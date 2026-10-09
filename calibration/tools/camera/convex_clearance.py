"""Independent separating-plane certificates for degenerate mesh queries.

MuJoCo sometimes returns zero for separated convex meshes. FCL supplies a
candidate axis; extrema of the actual compiled vertices independently certify
the separation. No mesh pair is excluded and overlaps cannot gain clearance.
"""
import numpy as np


class ConvexSeparation:
    def __init__(self, model):
        self.model = model
        self.cache = {}

    def shape(self, geom):
        import hppfcl
        import mujoco
        from scipy.spatial import ConvexHull

        if geom in self.cache:
            return self.cache[geom]
        model = self.model
        kind = model.geom_type[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start + count].astype(float)
            hull = ConvexHull(vertices)
            indices = hull.vertices
            mapping = {int(k): i for i, k in enumerate(indices)}
            points, triangles = hppfcl.StdVec_Vec3f(), hppfcl.StdVec_Triangle()
            for point in vertices[indices]:
                points.append(point)
            for face, equation in zip(hull.simplices, hull.equations):
                face = face.copy()
                normal = np.cross(vertices[face[1]] - vertices[face[0]],
                                  vertices[face[2]] - vertices[face[0]])
                if normal @ equation[:3] < 0:
                    face[[1, 2]] = face[[2, 1]]
                triangles.append(hppfcl.Triangle(*[mapping[int(k)] for k in face]))
            shape = hppfcl.Convex(points, triangles)
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            size = model.geom_size[geom]
            vertices = np.array([[x, y, z] for x in (-size[0], size[0])
                                 for y in (-size[1], size[1])
                                 for z in (-size[2], size[2])])
            shape = hppfcl.Box(2 * size)
        else:
            return None
        self.cache[geom] = shape, vertices
        return self.cache[geom]

    def certificate(self, data, first, second):
        import hppfcl

        a, b = self.shape(first), self.shape(second)
        if a is None or b is None:
            return 0.
        rotations = [data.geom_xmat[g].reshape(3, 3) for g in (first, second)]
        positions = [data.geom_xpos[g] for g in (first, second)]
        transforms = [hppfcl.Transform3f(r, p) for r, p in zip(rotations, positions)]
        request, result = hppfcl.DistanceRequest(), hppfcl.DistanceResult()
        request.enable_nearest_points = True
        distance = hppfcl.distance(a[0], transforms[0], b[0], transforms[1], request, result)
        if not np.isfinite(distance) or distance <= 0:
            return 0.
        axis = np.asarray(result.getNearestPoint2()) - np.asarray(result.getNearestPoint1())
        norm = np.linalg.norm(axis)
        if not np.isfinite(norm) or norm < 1e-10:
            return 0.
        axis /= norm
        projections = [vertices @ (rotation.T @ axis) + position @ axis
                       for (_, vertices), rotation, position in zip((a, b), rotations, positions)]
        # Any positive projection gap is a lower bound on Euclidean distance
        # for every vertex and its convex combinations, independent of FCL.
        gap = max(projections[1].min() - projections[0].max(),
                  projections[0].min() - projections[1].max())
        return max(0., float(gap) - 1e-7)
