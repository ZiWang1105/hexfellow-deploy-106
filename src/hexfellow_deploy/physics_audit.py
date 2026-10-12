"""Plant-only evidence accumulated at every physics step, never a control input."""
import json
from pathlib import Path

import numpy as np


class PhysicsAudit:
    """Audit contacts and generic 10 cm pickup retention without selecting a task."""
    def __init__(self, model, data):
        from hexfellow_deploy.sim.contacts import is_static_body
        self.model = model
        self.dt = float(model.opt.timestep)
        self.names = [model.body(body).name or 'world' for body in range(model.nbody)]
        self.entity = [name.startswith('entity_') for name in self.names]
        self.static = [self.entity[body] and is_static_body(model, body) for body in range(model.nbody)]
        self.robot = [body != 0 and not self.entity[body] for body in range(model.nbody)]
        self.geom_body = np.asarray(model.geom_bodyid).copy()
        self.jaw_side = {body: side for body, name in enumerate(self.names)
                         for side in ('left', 'right') if name.startswith(side+'_jaw_')}
        flex_bodies = set(model.flex_vertbodyid)
        self.objects = [body for body in range(model.nbody) if self.entity[body]
                        and model.body_jntnum[body] > 0 and body not in flex_bodies]
        self.initial = {body: data.subtree_com[body].copy() for body in self.objects}
        self.tcp = {side: model.site(side+'_tcp').id for side in ('left', 'right')}
        self.since = {}
        self.last_objects = []
        self.samples = 0
        self.initial_time_s = float(data.time)
        self.first_s = None
        self.last_s = None
        self.max_interval_s = 0.
        self.contact_steps = 0
        self.first_contact = None
        self.groups = {}

    def step(self, data):
        now = float(data.time)
        if self.last_s is not None:
            if now <= self.last_s:
                raise ValueError('Physics audit requires a new physics step')
            interval = now-self.last_s
            self.max_interval_s = max(self.max_interval_s, interval)
            if interval > self.dt*1.5:
                # A missing physics sample cannot extend verified retention.
                self.since.clear()
        if self.first_s is None:
            self.first_s = now
            self.max_interval_s = max(self.max_interval_s, now-self.initial_time_s)
        self.last_s = now
        self.samples += 1
        touching = {body: {'left': set(), 'right': set()} for body in self.objects}
        supported = set()
        forbidden = set()
        for contact in data.contact:
            if contact.dist > 0 or contact.geom1 < 0 or contact.geom2 < 0:
                continue
            a, b = int(self.geom_body[contact.geom1]), int(self.geom_body[contact.geom2])
            for body, other in ((a, b), (b, a)):
                if self.static[body] and self.robot[other]:
                    forbidden.add((self.names[body], self.names[other]))
                if body in touching:
                    side = self.jaw_side.get(other)
                    if side is not None:
                        touching[body][side].add(other)
                    if other == 0 or self.entity[other]:
                        supported.add(body)
        if forbidden:
            pairs = tuple(sorted(forbidden))
            self.contact_steps += 1
            if self.first_contact is None:
                self.first_contact = dict(time_s=now, bodies=pairs)
            group = self.groups.setdefault(pairs, dict(steps=0, first_s=now, last_s=now))
            group['steps'] += 1
            group['last_s'] = now
        rows = []
        for body in self.objects:
            com = data.subtree_com[body]
            height = float(com[2]-self.initial[body][2])
            arms = [side for side, jaws in touching[body].items() if len(jaws) == 2
                    and np.linalg.norm(com-data.site_xpos[self.tcp[side]]) < .09]
            retained = height >= .10 and bool(arms) and body not in supported
            if retained:
                self.since.setdefault(body, now)
            else:
                self.since.pop(body, None)
            rows.append(dict(name=self.names[body], com_m=com.tolist(), height_gain_m=height,
                grasping_arms=arms, opposing_jaw_contacts=any(len(jaws) == 2 for jaws in touching[body].values()),
                support_contact=body in supported, retained_s=now-self.since.get(body, now)))
        self.last_objects = rows

    def report(self):
        return dict(schema=1, sampling='every_physics_step', timestep_s=self.dt,
            samples=self.samples, initial_time_s=self.initial_time_s, first_time_s=self.first_s, time_s=self.last_s,
            max_sample_interval_s=self.max_interval_s,
            complete=bool(self.samples and self.max_interval_s <= self.dt*1.5),
            robot_static_contact_steps=self.contact_steps, first_robot_static_contact=self.first_contact,
            contact_groups=[dict(bodies=pairs, **details) for pairs, details in sorted(self.groups.items())],
            objects=self.last_objects,
            criterion='10 cm COM rise, opposing jaws of one hand, no support, uninterrupted sampled retention',
            limitation='Discrete physics-step evidence; no assertion about contacts between integration steps')

    def save(self, directory):
        path = Path(directory)/'physics_evaluation.json'
        path.write_text(json.dumps(self.report(), indent=2)+'\n')
        return path


def pickup_evaluation(audit, body, *, final_hold, completed_budget):
    """Called only after shutdown; audited state never feeds policy or control."""
    retained = any(item['name'] == body and item['retained_s'] >= 10. for item in audit['objects'])
    contacts = int(audit['robot_static_contact_steps'])
    return dict(retained_pickup=retained, robot_static_contact_steps=contacts,
        final_hold_verified=bool(final_hold), evaluated_body=body,
        sampling=audit['sampling'], timestep_s=audit['timestep_s'], samples=audit['samples'],
        complete=audit['complete'], first_robot_static_contact=audit['first_robot_static_contact'],
        accepted=bool(retained and contacts == 0 and audit['complete'] and final_hold and completed_budget))
