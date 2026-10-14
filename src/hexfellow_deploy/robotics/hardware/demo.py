"""Shared research acceptance and stop-result predicates."""
ACCEPTED = ("drive_protection_checked", "disconnect_stop_checked", "sdk_startup_checked", "initialization_stop_checked")


def failure_detail(result):
    outcome = result.get('gateway', {})
    details = [f"Main result:{result.get('finish_reason', 'unknown')}"]
    if outcome.get('stop_failures'):
        details.append('Shutdown:' + '; '.join(outcome['stop_failures']))
    for error in (outcome.get('diagnostics', {}).get('error'), result.get('fault_evidence_error')):
        if error:
            details.append(str(error))
    return '；'.join(details)


def machine_pass(name, report):
    if not report.get("passed") or report.get("trace_error") or report.get("trace_dropped_rows"):
        return False
    if report.get("mode") != "live" or not report.get("demo_authorized") or not report.get("motion_commands_sent"):
        return False
    gateway = report.get("gateway", {})
    if gateway.get("kind") != "closed":
        return False
    if gateway.get('stop_failures') or report.get('fault_evidence_error'):
        return False
    if gateway.get('diagnostics', {}).get('complete') is False:
        return False
    phases = [r["phase"] for r in report.get("phases", [])]
    if name == "body":
        motion = 'body-check' in phases
        return (report.get("estop_observed") is True and motion and "stopping" in phases
                and report.get('finish_reason') == 'expected_physical_estop'
                and all(gateway.get('stop_observations', {}).get(n, {}).get('continuous_stop_confirmed')
                        for n in ('left', 'right', 'theta', 'base')))
    if gateway.get("fault") or gateway.get("stop_failures"):
        return False
    if name == "replay":
        return {"default", "init", "ready", "replay", "settle", "hold"}.issubset(phases)
    return (name == "quest" and report.get("quest_active_cycles", 0) >= 50
            and report.get("quest_release_events", 0) >= 1 and "stopping" in phases)
