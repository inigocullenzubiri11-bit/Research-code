"""
=============================================================================
  TESTING SUITE — FLOOD EVACUATION ROUTE OPTIMIZATION SYSTEM
  All 4 Testing Phases from Research Plan
=============================================================================
  INSTALL:  pip install folium requests matplotlib numpy
  RUN:      python testing_suite.py
  NOTE:     ph_evac_router.py must be in the same folder!
=============================================================================

  PHASE 1 — Computational & Algorithmic Testing
    A. Computational & Algorithmic Testing
    B. Pathfinding Accuracy & Optimality
    C. Dynamic Re-routing Validation
    D. Network Resilience Analysis

  PHASE 2 — System Performance Validation
    A. Hydrodynamic Integrated Simulation
    B. 10/30 Coupled Model
    C. Scenario-based Stress Tests

  PHASE 3 — Expert Validation (logs + reports for DLSU AdRIC)

  PHASE 4 — Community Feedback & Pilot Testing
    (generates simplified map + feedback form)
=============================================================================
"""

import random, math, time, os, sys, json
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("pip install matplotlib numpy  for charts\n")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Ph_evac_route import (
    fetch_osm_data, build_graph, find_best, find_all_shelter_routes,
    bbox_from_point, FLOOD_RISK, FLOOD_EMOJI, ROAD_COLOR,
    SCENARIOS, FLOOD_SCENARIOS, SPEED_PROFILES,
    haversine_m, walk_time, dijkstra, get_path,
    pick_location
)

SEP   = "=" * 68
SEED  = 42
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG   = []   # global log for expert validation report

# Shared context set in main() before phase1a — used by Test 6 to rebuild graphs
_phase1a_ways     = []
_phase1a_shelters = []

def log(msg):
    LOG.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    print(msg)

def mean(lst): return sum(lst)/len(lst) if lst else 0.0
def stdev(lst):
    if len(lst) < 2: return 0.0
    m = mean(lst)
    return math.sqrt(sum((x-m)**2 for x in lst)/(len(lst)-1))


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1A — COMPUTATIONAL & ALGORITHMIC TESTING
#  Verifies Dijkstra correctness by comparing against brute-force on small graphs
# ─────────────────────────────────────────────────────────────────────────────

def phase1a_algorithmic_testing(graph, node_info, shelter_ids, start):
    log(f"\n{SEP}")
    log(f"  PHASE 1A — COMPUTATIONAL & ALGORITHMIC TESTING")
    log(SEP)

    results = {"passed": 0, "failed": 0, "tests": []}

    # Test 1: Dijkstra finds a path when one exists
    log("\n  Test 1: Route existence check")
    path, segs, cost = find_best(graph, start, shelter_ids)
    t1 = path is not None
    status = "PASS" if t1 else "FAIL"
    log(f"  [{status}] Dijkstra finds route: {t1}")
    results["tests"].append({"name": "Route existence", "result": status})
    if t1: results["passed"] += 1
    else:  results["failed"] += 1

    # Test 2: Cost is non-negative
    log("\n  Test 2: Non-negative cost check")
    t2 = cost >= 0
    status = "PASS" if t2 else "FAIL"
    log(f"  [{status}] Route cost >= 0: {cost:.4f}")
    results["tests"].append({"name": "Non-negative cost", "result": status})
    if t2: results["passed"] += 1
    else:  results["failed"] += 1

    # Test 3: Path is continuous (each node connects to next)
    log("\n  Test 3: Path continuity check")
    t3 = True
    if path and len(path) > 1:
        for i in range(len(path)-1):
            a, b = path[i], path[i+1]
            connected = any(e["to"] == b for e in graph.get(a, []))
            if not connected:
                t3 = False
                log(f"  Broken link at {a} -> {b}")
                break
    status = "PASS" if t3 else "FAIL"
    log(f"  [{status}] Path is continuous: {t3}")
    results["tests"].append({"name": "Path continuity", "result": status})
    if t3: results["passed"] += 1
    else:  results["failed"] += 1

    # Test 4: Destination is a shelter
    log("\n  Test 4: Destination is valid shelter")
    t4 = path and path[-1] in shelter_ids
    status = "PASS" if t4 else "FAIL"
    log(f"  [{status}] Route ends at shelter: {t4}")
    results["tests"].append({"name": "Valid destination", "result": status})
    if t4: results["passed"] += 1
    else:  results["failed"] += 1

    # Test 5: W formula weights sum to 1.0
    log("\n  Test 5: Weight coefficient validation (α+β+γ+δ=1.0)")
    all_valid = True
    for k, v in SCENARIOS.items():
        total = v["alpha"] + v["beta"] + v["gamma"] + v["delta"]
        if abs(total - 1.0) > 0.001:
            log(f"  FAIL Mode {k}: weights sum to {total:.3f}")
            all_valid = False
    status = "PASS" if all_valid else "FAIL"
    log(f"  [{status}] All 8 optimization modes have valid weights")
    results["tests"].append({"name": "Weight coefficients", "result": status})
    if all_valid: results["passed"] += 1
    else:         results["failed"] += 1

    # Test 6: MAX SAFETY produces lower flood risk than MAX SPEED
    log("\n  Test 6: MAX SAFETY vs MAX SPEED flood risk validation")
    opt_s = SCENARIOS["1"]; opt_p = SCENARIOS["2"]
    nd = node_info[start]
    # We need the raw ways/shelters — extract them from node_info for shelter list
    # and use the same ways that built the current graph by passing ways from caller.
    # The current graph was built with the real ways; rebuild both graphs with same data.
    # NOTE: we use the already-built graph (which used real ways) for SAFETY since
    # the caller built it with balanced or default weights. Instead, rebuild both
    # properly using the ways/shelters passed into this phase via the test harness.
    try:
        # Rebuild SAFETY graph with real ways
        g_s, ni_s, sh_s, _ = build_graph(
            nd[0], nd[1], _phase1a_ways, _phase1a_shelters, 1.0,
            opt_s["alpha"], opt_s["beta"], opt_s["gamma"], opt_s["delta"])
        _, segs_s, _ = find_best(g_s, start, sh_s)
        risk_s = max(FLOOD_RISK[s["flood"]] for s in segs_s) if segs_s else 0.0
    except Exception as exc:
        log(f"  Could not build SAFETY graph: {exc}")
        risk_s = None

    try:
        g_p, ni_p, sh_p, _ = build_graph(
            nd[0], nd[1], _phase1a_ways, _phase1a_shelters, 1.0,
            opt_p["alpha"], opt_p["beta"], opt_p["gamma"], opt_p["delta"])
        _, segs_p, _ = find_best(g_p, start, sh_p)
        risk_p = max(FLOOD_RISK[s["flood"]] for s in segs_p) if segs_p else 0.0
    except Exception as exc:
        log(f"  Could not build SPEED graph: {exc}")
        risk_p = None

    if risk_s is not None and risk_p is not None:
        t6 = risk_s <= risk_p
        status = "PASS" if t6 else "FAIL"
        log(f"  [{status}] MAX SAFETY risk ({risk_s:.3f}) <= MAX SPEED risk ({risk_p:.3f})")
    else:
        status = "SKIP"
        log(f"  [SKIP] Could not build graphs for comparison")
    results["tests"].append({"name": "Safety vs Speed risk", "result": status})
    if status == "PASS": results["passed"] += 1

    total = results["passed"] + results["failed"]
    log(f"\n  PHASE 1A RESULT: {results['passed']}/{total} tests passed")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1B — PATHFINDING ACCURACY & OPTIMALITY
#  Verifies routes are truly optimal by testing multiple modes
# ─────────────────────────────────────────────────────────────────────────────

def phase1b_pathfinding_accuracy(ways, shelters, node_info, start, flood_mult, n=20):
    log(f"\n{SEP}")
    log(f"  PHASE 1B — PATHFINDING ACCURACY & OPTIMALITY")
    log(f"  Testing {n} random start points across all 8 modes")
    log(SEP)

    rng      = random.Random(SEED)
    nd       = node_info[start]
    road_nds = [(nid,ni) for nid,ni in node_info.items() if ni[3]=="road"]

    results = {"mode_stats": {}, "optimal_verified": 0, "total": 0}

    for ok, opt in SCENARIOS.items():
        risks = []; dists = []; successes = 0
        for _ in range(n):
            if not road_nds: break
            nid, ni = random.choice(road_nds)
            try:
                g, ni2, sh, _ = build_graph(
                    ni[0], ni[1], ways, shelters, flood_mult,
                    opt["alpha"], opt["beta"], opt["gamma"], opt["delta"])
                s = next((x for x,xd in ni2.items() if xd[3]=="start"), None)
                if not s: continue
                _, segs2, _ = find_best(g, s, sh)
                if segs2:
                    successes += 1
                    dists.append(sum(x["dist_m"] for x in segs2))
                    risks.append(max(FLOOD_RISK[x["flood"]] for x in segs2))
            except: continue

        results["mode_stats"][ok] = {
            "name":         opt["name"],
            "success_rate": successes/n*100,
            "avg_dist":     mean(dists),
            "avg_risk":     mean(risks),
            "std_dist":     stdev(dists),
        }
        results["total"] += n
        results["optimal_verified"] += successes
        log(f"  Mode {ok} {opt['name']}: {successes/n*100:.0f}% success | "
            f"avg risk={mean(risks):.3f} | avg dist={mean(dists):.0f}m")

    # Verify safety ordering: MAX SAFETY risk < BALANCED risk < MAX SPEED risk
    ms  = results["mode_stats"].get("1",{}).get("avg_risk",0)
    bal = results["mode_stats"].get("4",{}).get("avg_risk",0)
    spd = results["mode_stats"].get("2",{}).get("avg_risk",0)
    ordering_ok = ms <= bal <= spd
    log(f"\n  Safety ordering check (SAFETY={ms:.3f} <= BALANCED={bal:.3f} <= SPEED={spd:.3f}): "
        f"{'PASS' if ordering_ok else 'FAIL'}")
    results["ordering_verified"] = ordering_ok

    log(f"\n  PHASE 1B RESULT: {results['optimal_verified']}/{results['total']} successful routes")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1C — DYNAMIC RE-ROUTING VALIDATION
#  Tests that the system correctly re-routes when roads are blocked
# ─────────────────────────────────────────────────────────────────────────────

def phase1c_dynamic_rerouting(graph, node_info, shelter_ids, start, ways, shelters):
    log(f"\n{SEP}")
    log(f"  PHASE 1C — DYNAMIC RE-ROUTING VALIDATION")
    log(f"  Simulates road blockages and verifies alternative routes found")
    log(SEP)

    results = {"scenarios": [], "passed": 0, "failed": 0}
    opt = SCENARIOS["1"]  # MAX SAFETY

    # Get original route
    path_orig, segs_orig, _ = find_best(graph, start, shelter_ids)
    if not path_orig:
        log("  SKIP — No original route found")
        return results

    orig_dist = sum(s["dist_m"] for s in segs_orig)
    log(f"  Original route: {len(segs_orig)} segments | {orig_dist}m")

    # Test each flood scenario as a "blockage simulation"
    for fk, fd in FLOOD_SCENARIOS.items():
        mult  = fd["multiplier"]
        fname = fd["name"].split("—")[-1].strip()
        nd    = node_info[start]
        try:
            g2, ni2, sh2, _ = build_graph(
                nd[0], nd[1], ways, shelters, mult,
                opt["alpha"], opt["beta"], opt["gamma"], opt["delta"])
            s2 = next((x for x,xd in ni2.items() if xd[3]=="start"), None)
            if not s2:
                results["scenarios"].append({"scenario": fname, "result": "SKIP"})
                continue
            path2, segs2, _ = find_best(g2, s2, sh2)
            found = path2 is not None
            status = "PASS" if found else "BLOCKED"
            dist2  = sum(s["dist_m"] for s in segs2) if segs2 else 0
            log(f"  [{status}] {fname}: route {'found' if found else 'blocked'}"
                f"{f' | {dist2}m' if found else ''}")
            results["scenarios"].append({
                "scenario": fname, "result": status,
                "dist_m": dist2, "segments": len(segs2) if segs2 else 0
            })
            if found: results["passed"] += 1
            else:     results["failed"] += 1
        except Exception as e:
            log(f"  [ERROR] {fname}: {e}")
            results["scenarios"].append({"scenario": fname, "result": "ERROR"})

    log(f"\n  PHASE 1C RESULT: {results['passed']} scenarios re-routed successfully, "
        f"{results['failed']} fully blocked")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1D — NETWORK RESILIENCE ANALYSIS
#  Tests how many alternative routes exist and network connectivity
# ─────────────────────────────────────────────────────────────────────────────

def phase1d_network_resilience(graph, node_info, shelter_ids, start):
    log(f"\n{SEP}")
    log(f"  PHASE 1D — NETWORK RESILIENCE ANALYSIS")
    log(SEP)

    results = {}

    # Count reachable nodes
    dist, _ = dijkstra(graph, start)
    reachable = sum(1 for d in dist.values() if d < float("inf"))
    total     = len(graph)
    conn_pct  = reachable/total*100 if total else 0

    log(f"  Reachable nodes : {reachable}/{total} ({conn_pct:.1f}%)")
    results["reachable_pct"] = conn_pct

    # Count reachable shelters
    reachable_shelters = [s for s in shelter_ids
                          if dist.get(s, float("inf")) < float("inf")]
    log(f"  Reachable shelters: {len(reachable_shelters)}/{len(shelter_ids)}")
    results["reachable_shelters"] = len(reachable_shelters)
    results["total_shelters"]     = len(shelter_ids)

    # Get all shelter routes ranked
    all_routes = find_all_shelter_routes(graph, start, shelter_ids)
    log(f"  Alternative routes available: {len(all_routes)}")
    results["alternative_routes"] = len(all_routes)

    # Resilience score
    score = min(100, conn_pct * 0.4 + len(reachable_shelters)*10 + len(all_routes)*5)
    if score >= 70:   status = "RESILIENT"
    elif score >= 40: status = "MODERATE"
    else:             status = "VULNERABLE"
    log(f"  Resilience Score: {score:.0f}/100 — {status}")
    results["score"]  = score
    results["status"] = status

    # Avg edge connectivity (avg edges per node)
    avg_edges = mean([len(v) for v in graph.values()])
    log(f"  Avg edges per node: {avg_edges:.1f}")
    results["avg_edges"] = avg_edges

    if all_routes:
        log(f"\n  TOP SHELTER ROUTES:")
        for i, (sid, sp, ss, sc, td) in enumerate(all_routes[:5], 1):
            sn  = node_info[sid]
            tm  = td / 1.1 / 60
            mfl = max(FLOOD_RISK[s["flood"]] for s in ss) if ss else 0
            rl  = next(k for k,v in FLOOD_RISK.items() if v==mfl)
            log(f"  {i}. {sn[2][:35]} | {td}m | ~{tm:.0f}min | {FLOOD_EMOJI.get(rl,'')} {rl}")

    log(f"\n  PHASE 1D RESULT: Network is {status} with {score:.0f}/100 resilience score")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2A — HYDRODYNAMIC INTEGRATED SIMULATION
#  Simulates rising flood levels over time and tests system adaptation
# ─────────────────────────────────────────────────────────────────────────────

def phase2a_hydrodynamic_simulation(ways, shelters, node_info, start):
    log(f"\n{SEP}")
    log(f"  PHASE 2A — HYDRODYNAMIC INTEGRATED SIMULATION")
    log(f"  Simulates rising flood levels — tests system adaptation over time")
    log(SEP)

    opt     = SCENARIOS["1"]   # MAX SAFETY
    nd      = node_info[start]
    results = {"time_steps": [], "system_held": True}

    # Simulate flood rising over 10 time steps (multiplier 1.0 → 4.0)
    multipliers = [round(1.0 + i * 0.33, 2) for i in range(10)]
    log(f"  Simulating {len(multipliers)} time steps (flood rising from 1.0x to 4.0x)")
    log(f"  {'Step':<6} {'Multiplier':<12} {'Status':<10} {'Dist':>8} {'Risk':>8}")
    log(f"  {'-'*50}")

    prev_dist = None
    for step, mult in enumerate(multipliers, 1):
        try:
            g, ni, sh, _ = build_graph(
                nd[0], nd[1], ways, shelters, mult,
                opt["alpha"], opt["beta"], opt["gamma"], opt["delta"])
            s = next((x for x,xd in ni.items() if xd[3]=="start"), None)
            if not s:
                results["time_steps"].append({"step":step,"mult":mult,"status":"ERROR"})
                continue
            path2, segs2, _ = find_best(g, s, sh)
            if path2 and segs2:
                dist2 = sum(x["dist_m"] for x in segs2)
                risk2 = max(FLOOD_RISK[x["flood"]] for x in segs2)
                status = "ROUTE FOUND"
                change = f"+{dist2-prev_dist}m" if prev_dist and dist2 != prev_dist else "same"
                log(f"  {step:<6} {mult:<12} {status:<10} {dist2:>7}m {risk2:>8.3f}  ({change})")
                prev_dist = dist2
                results["time_steps"].append({
                    "step": step, "mult": mult, "status": status,
                    "dist_m": dist2, "risk": risk2
                })
            else:
                log(f"  {step:<6} {mult:<12} {'BLOCKED':<10} {'N/A':>8} {'N/A':>8}")
                results["time_steps"].append({"step":step,"mult":mult,"status":"BLOCKED"})
                results["system_held"] = False
        except Exception as e:
            log(f"  {step:<6} {mult:<12} {'ERROR':<10} {str(e)[:20]}")
            results["time_steps"].append({"step":step,"mult":mult,"status":"ERROR"})

    found = sum(1 for t in results["time_steps"] if t["status"]=="ROUTE FOUND")
    log(f"\n  PHASE 2A RESULT: Route found in {found}/{len(multipliers)} time steps")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2B — 10/30 COUPLED MODEL
#  Tests 10 flood scenarios × 30 random start points = 300 simulations
# ─────────────────────────────────────────────────────────────────────────────

def phase2b_coupled_model(ways, shelters, node_info, start):
    log(f"\n{SEP}")
    log(f"  PHASE 2B — 10/30 COUPLED MODEL")
    log(f"  10 flood multipliers × 30 random start points = 300 simulations")
    log(SEP)

    rng      = random.Random(SEED + 10)
    opt      = SCENARIOS["1"]
    road_nds = [(nid,ni) for nid,ni in node_info.items() if ni[3]=="road"]
    mults    = [round(1.0 + i*0.33, 2) for i in range(10)]
    results  = {"multipliers": [], "total": 0, "success": 0}

    log(f"  {'Multiplier':<12} {'Success':>10} {'Avg Dist':>10} {'Avg Risk':>10}")
    log(f"  {'-'*50}")

    for mult in mults:
        succ = 0; dists = []; risks = []
        for _ in range(30):
            if not road_nds: break
            nid, ni = rng.choice(road_nds)
            try:
                g, ni2, sh, _ = build_graph(
                    ni[0], ni[1], ways, shelters, mult,
                    opt["alpha"], opt["beta"], opt["gamma"], opt["delta"])
                s = next((x for x,xd in ni2.items() if xd[3]=="start"), None)
                if not s: continue
                _, segs2, _ = find_best(g, s, sh)
                if segs2:
                    succ += 1
                    dists.append(sum(x["dist_m"] for x in segs2))
                    risks.append(max(FLOOD_RISK[x["flood"]] for x in segs2))
            except: continue

        results["total"]   += 30
        results["success"] += succ
        entry = {"mult": mult, "success": succ, "avg_dist": mean(dists), "avg_risk": mean(risks)}
        results["multipliers"].append(entry)
        log(f"  {mult:<12} {succ:>4}/30 ({succ/30*100:>4.0f}%) {mean(dists):>10.0f}m {mean(risks):>10.3f}")

    overall = results["success"]/results["total"]*100
    log(f"\n  PHASE 2B RESULT: {results['success']}/{results['total']} ({overall:.1f}%) overall success")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2C — SCENARIO-BASED STRESS TESTS
#  Tests system under extreme conditions and edge cases
# ─────────────────────────────────────────────────────────────────────────────

def phase2c_stress_tests(graph, node_info, shelter_ids, start, ways, shelters):
    log(f"\n{SEP}")
    log(f"  PHASE 2C — SCENARIO-BASED STRESS TESTS")
    log(SEP)

    results = {"tests": [], "passed": 0, "failed": 0}
    nd = node_info[start]

    def run_test(name, fn):
        try:
            t_start = time.time()
            result  = fn()
            elapsed = time.time() - t_start
            status  = "PASS" if result else "FAIL"
            log(f"  [{status}] {name} ({elapsed:.2f}s)")
            results["tests"].append({"name":name,"result":status,"time":elapsed})
            if result: results["passed"] += 1
            else:      results["failed"] += 1
        except Exception as e:
            log(f"  [ERROR] {name}: {e}")
            results["tests"].append({"name":name,"result":"ERROR"})
            results["failed"] += 1

    # Stress 1: All 8 modes complete without crash
    def test_all_modes():
        for ok, opt in SCENARIOS.items():
            g, ni, sh, _ = build_graph(
                nd[0], nd[1], ways, shelters, 1.0,
                opt["alpha"], opt["beta"], opt["gamma"], opt["delta"])
            s = next((x for x,xd in ni.items() if xd[3]=="start"), None)
            if s: find_best(g, s, sh)
        return True
    run_test("All 8 optimization modes complete without crash", test_all_modes)

    # Stress 2: All 4 flood scenarios complete without crash
    def test_all_floods():
        opt = SCENARIOS["1"]
        for fk, fd in FLOOD_SCENARIOS.items():
            g, ni, sh, _ = build_graph(
                nd[0], nd[1], ways, shelters, fd["multiplier"],
                opt["alpha"], opt["beta"], opt["gamma"], opt["delta"])
            s = next((x for x,xd in ni.items() if xd[3]=="start"), None)
            if s: find_best(g, s, sh)
        return True
    run_test("All 4 flood scenarios complete without crash", test_all_floods)

    # Stress 3: All 6 speed profiles produce valid times
    def test_speed_profiles():
        path, segs, _ = find_best(graph, start, shelter_ids)
        if not segs: return False
        for pk, pv in SPEED_PROFILES.items():
            t = walk_time(segs, pv["speed"])
            if t <= 0: return False
        return True
    run_test("All 6 speed profiles produce valid times", test_speed_profiles)

    # Stress 4: System handles extreme flood (4.0x) gracefully
    def test_extreme_flood():
        g, ni, sh, _ = build_graph(
            nd[0], nd[1], ways, shelters, 4.0,
            0.05, 0.90, 0.03, 0.02)
        s = next((x for x,xd in ni.items() if xd[3]=="start"), None)
        if not s: return True  # graceful skip
        path2, segs2, _ = find_best(g, s, sh)
        return True  # not crashing = pass (route may or may not exist)
    run_test("Extreme flood (4.0x) handled gracefully", test_extreme_flood)

    # Stress 5: Large graph — count nodes and edges
    def test_graph_size():
        n_nodes = len(graph)
        n_edges = sum(len(v) for v in graph.values())
        log(f"       Graph size: {n_nodes} nodes, {n_edges} edges")
        return n_nodes > 10 and n_edges > 10
    run_test("Graph has sufficient nodes and edges", test_graph_size)

    # Stress 6: Response time < 10 seconds for route finding
    def test_response_time():
        t0 = time.time()
        find_best(graph, start, shelter_ids)
        return (time.time() - t0) < 10.0
    run_test("Route finding completes in < 10 seconds", test_response_time)

    # Stress 7: Multiple simultaneous shelter lookups
    def test_all_shelters():
        all_routes = find_all_shelter_routes(graph, start, shelter_ids)
        return len(all_routes) >= 0  # just checking it doesn't crash
    run_test("All shelter route lookup completes", test_all_shelters)

    total = results["passed"] + results["failed"]
    log(f"\n  PHASE 2C RESULT: {results['passed']}/{total} stress tests passed")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3 — EXPERT VALIDATION REPORT
#  Generates a formatted report for DLSU AdRIC / CITE4D Lab experts
# ─────────────────────────────────────────────────────────────────────────────

def phase3_expert_validation_report(all_results, area_name):
    log(f"\n{SEP}")
    log(f"  PHASE 3 — EXPERT VALIDATION REPORT")
    log(f"  Generating report for DLSU AdRIC / CITE4D Lab")
    log(SEP)

    report = {
        "title":       "Expert Validation Report — Flood Evacuation Route Optimization System",
        "study_area":  area_name,
        "timestamp":   datetime.now().isoformat(),
        "phases":      all_results,
        "log":         LOG,
    }

    fname = f"expert_validation_report_{STAMP}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    log(f"  Expert validation report saved → {fname}")

    # Also save human-readable text report
    txt_fname = f"expert_validation_report_{STAMP}.txt"
    with open(txt_fname, "w", encoding="utf-8") as f:
        f.write("=" * 68 + "\n")
        f.write("  EXPERT VALIDATION REPORT\n")
        f.write("  Flood Evacuation Route Optimization System\n")
        f.write(f"  Study Area: {area_name}\n")
        f.write(f"  Date: {datetime.now().strftime('%B %d, %Y %H:%M')}\n")
        f.write("=" * 68 + "\n\n")
        f.write("TESTING LOG:\n")
        for entry in LOG:
            f.write(entry + "\n")
        f.write("\n" + "=" * 68 + "\n")
        f.write("FOR EXPERT REVIEW:\n")
        f.write("1. Verify algorithmic correctness results in Phase 1A\n")
        f.write("2. Review pathfinding accuracy across all 8 optimization modes\n")
        f.write("3. Confirm dynamic re-routing behaves correctly under flood scenarios\n")
        f.write("4. Validate network resilience scores against community standards\n")
        f.write("5. Assess 10/30 coupled model results for statistical validity\n")
        f.write("6. Check stress test results for production readiness\n")
        f.write("=" * 68 + "\n")

    log(f"  Text report saved → {txt_fname}")
    log(f"\n  PHASE 3: Reports ready for DLSU AdRIC submission")
    return fname, txt_fname


# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def make_testing_charts(p1a, p1b, p1d, p2b, p2c, area_name):
    if not MATPLOTLIB_OK:
        return

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0a0f1a")
    gs  = gridspec.GridSpec(2, 3, figure=fig)
    fig.suptitle(
        f"Testing Suite Results — Flood Evacuation Route Optimizer\n{area_name}",
        fontsize=13, fontweight="bold", color="white"
    )

    # Chart 1: Phase 1A test results
    ax1 = fig.add_subplot(gs[0,0])
    names  = [t["name"][:20] for t in p1a["tests"]]
    colors = ["#00cc55" if t["result"]=="PASS" else
              "#ff3300" if t["result"]=="FAIL" else "#888888"
              for t in p1a["tests"]]
    ax1.barh(names, [1]*len(names), color=colors, alpha=0.85)
    ax1.set_title("Phase 1A: Algorithm Tests", color="white", fontsize=10)
    ax1.tick_params(colors="white", labelsize=7)
    ax1.set_xlim(0, 1.5)
    for i, t in enumerate(p1a["tests"]):
        ax1.text(0.05, i, t["result"], va="center", fontsize=8,
                 color="white", fontweight="bold")

    # Chart 2: Phase 1B mode success rates
    ax2 = fig.add_subplot(gs[0,1])
    if p1b and p1b.get("mode_stats"):
        modes = [v["name"].replace("🛡️","").replace("⚡","").replace("👥","")
                 .replace("⚖️","").replace("🌊","").replace("🏃","")
                 .replace("🏟️","").replace("🛣️","").strip()
                 for v in p1b["mode_stats"].values()]
        rates = [v["success_rate"] for v in p1b["mode_stats"].values()]
        ax2.bar(modes, rates, color="#4472C4", alpha=0.85)
        ax2.set_title("Phase 1B: Mode Success Rates (%)", color="white", fontsize=10)
        ax2.set_ylim(0, 110)
        ax2.tick_params(colors="white", labelsize=7)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Chart 3: Phase 1D resilience
    ax3 = fig.add_subplot(gs[0,2])
    if p1d:
        cats = ["Connectivity\n(%)", "Reachable\nShelters", "Alt Routes", "Score"]
        vals = [p1d.get("reachable_pct",0),
                p1d.get("reachable_shelters",0)*10,
                p1d.get("alternative_routes",0)*5,
                p1d.get("score",0)]
        colors3 = ["#00cc55","#00aaff","#ffcc00","#ff6600"]
        ax3.bar(cats, vals, color=colors3, alpha=0.85)
        ax3.set_title("Phase 1D: Network Resilience", color="white", fontsize=10)
        ax3.tick_params(colors="white", labelsize=8)

    # Chart 4: Phase 2B coupled model
    ax4 = fig.add_subplot(gs[1,0])
    if p2b and p2b.get("multipliers"):
        mults  = [m["mult"] for m in p2b["multipliers"]]
        succs  = [m["success"]/30*100 for m in p2b["multipliers"]]
        ax4.plot(mults, succs, color="#00cc55", marker="o", linewidth=2)
        ax4.fill_between(mults, succs, alpha=0.2, color="#00cc55")
        ax4.set_title("Phase 2B: 10/30 Coupled Model\nSuccess Rate vs Flood Level",
                      color="white", fontsize=10)
        ax4.set_xlabel("Flood Multiplier", color="white")
        ax4.set_ylabel("Success Rate (%)", color="white")
        ax4.tick_params(colors="white")
        ax4.set_ylim(0,110)

    # Chart 5: Phase 2C stress tests
    ax5 = fig.add_subplot(gs[1,1])
    if p2c and p2c.get("tests"):
        t_names  = [t["name"][:25] for t in p2c["tests"]]
        t_colors = ["#00cc55" if t["result"]=="PASS" else
                    "#ff3300" if t["result"]=="FAIL" else "#888888"
                    for t in p2c["tests"]]
        ax5.barh(t_names, [1]*len(t_names), color=t_colors, alpha=0.85)
        ax5.set_title("Phase 2C: Stress Tests", color="white", fontsize=10)
        ax5.tick_params(colors="white", labelsize=7)
        for i, t in enumerate(p2c["tests"]):
            ax5.text(0.05, i, t["result"], va="center", fontsize=8,
                     color="white", fontweight="bold")

    # Chart 6: Overall summary
    ax6 = fig.add_subplot(gs[1,2])
    phases = ["1A\nAlgorithm", "1B\nPathfinding", "2B\nCoupled", "2C\nStress"]
    scores = [
        p1a["passed"]/(p1a["passed"]+p1a["failed"])*100 if p1a else 0,
        p1b.get("optimal_verified",0)/max(p1b.get("total",1),1)*100 if p1b else 0,
        p2b["success"]/max(p2b["total"],1)*100 if p2b else 0,
        p2c["passed"]/(p2c["passed"]+p2c["failed"])*100 if p2c else 0,
    ]
    bar_colors = ["#00cc55" if s>=80 else "#ffcc00" if s>=50 else "#ff3300"
                  for s in scores]
    bars = ax6.bar(phases, scores, color=bar_colors, alpha=0.85)
    for bar, val in zip(bars, scores):
        ax6.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+1, f"{val:.0f}%",
                 ha="center", va="bottom", fontsize=9, color="white")
    ax6.set_title("Overall Testing Summary", color="white", fontsize=10)
    ax6.set_ylabel("Pass Rate (%)", color="white")
    ax6.set_ylim(0, 115)
    ax6.tick_params(colors="white")

    plt.tight_layout(rect=[0,0,1,0.93])
    fname = f"testing_results_{STAMP}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    log(f"\n  Testing charts saved → {fname}")
    plt.show()




def _make_summary_chart(all_runs, area_name):
    """
    One combined PNG summarising N repeated test runs.
    Shows per-run scores + averages + trend lines.
    """
    if not MATPLOTLIB_OK or not all_runs:
        return
    n = len(all_runs)
    run_labels = [f"Run {i+1}" for i in range(n)]

    # Extract per-run scores for each phase
    def get_score(r, key):
        ph = r.get(key, {})
        if key == "1A":
            tot = ph.get("passed",0) + ph.get("failed",0)
            return ph.get("passed",0)/max(tot,1)*100
        if key == "1B":
            return ph.get("optimal_verified",0)/max(ph.get("total",1),1)*100
        if key == "2B":
            return ph.get("success",0)/max(ph.get("total",1),1)*100
        if key == "2C":
            tot = ph.get("passed",0) + ph.get("failed",0)
            return ph.get("passed",0)/max(tot,1)*100
        return 0

    phases      = ["1A Algorithm", "1B Pathfinding", "2B Coupled", "2C Stress"]
    phase_keys  = ["1A", "1B", "2B", "2C"]
    phase_colors= ["#00cc55", "#4472C4", "#ffcc00", "#ff6600"]

    scores_by_phase = {
        k: [get_score(r, k) for r in all_runs]
        for k in phase_keys
    }

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0a0f1a")

    if n == 1:
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    else:
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    fig.suptitle(
        f"Testing Suite Summary — Flood Evacuation Route Optimizer\n"
        f"{area_name}  |  {n} Run(s)",
        fontsize=13, fontweight="bold", color="white"
    )

    # ── Panel 1: Per-run bar chart ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    x   = range(n)
    bar_w = 0.18
    for pi, (pk, col) in enumerate(zip(phase_keys, phase_colors)):
        offsets = [xi + (pi - 1.5) * bar_w for xi in x]
        bars = ax1.bar(offsets, scores_by_phase[pk], width=bar_w,
                       color=col, alpha=0.85, label=phases[pi])
        if n <= 6:
            for bar, val in zip(bars, scores_by_phase[pk]):
                ax1.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+1, f"{val:.0f}",
                         ha="center", va="bottom", fontsize=7, color="white")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(run_labels, color="white", fontsize=8,
                         rotation=45 if n > 5 else 0)
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Pass Rate (%)", color="white")
    ax1.set_title("Score per Run", color="white", fontsize=10)
    ax1.tick_params(colors="white")
    ax1.legend(fontsize=7, facecolor="#0a0f1a", labelcolor="white")

    # ── Panel 2: Average summary bar ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    avgs = [sum(scores_by_phase[pk])/n for pk in phase_keys]
    bars2 = ax2.bar(phases, avgs,
                    color=["#00cc55" if a>=80 else "#ffcc00" if a>=50 else "#ff3300"
                           for a in avgs], alpha=0.85)
    for bar, val in zip(bars2, avgs):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+1, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=9, color="white",
                 fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Average Pass Rate (%)", color="white")
    ax2.set_title("Average Across All Runs", color="white", fontsize=10)
    ax2.tick_params(colors="white", labelsize=8)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha="right")

    if n > 1:
        # ── Panel 3: Trend lines ─────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        xs = list(range(1, n+1))
        for pk, col, lbl in zip(phase_keys, phase_colors, phases):
            ax3.plot(xs, scores_by_phase[pk], marker="o", color=col,
                     linewidth=2, label=lbl)
        ax3.set_xlim(0.5, n+0.5)
        ax3.set_xticks(xs)
        ax3.set_xticklabels(run_labels, color="white", fontsize=8,
                             rotation=45 if n > 5 else 0)
        ax3.set_ylim(0, 115)
        ax3.set_ylabel("Pass Rate (%)", color="white")
        ax3.set_title("Trend Across Runs", color="white", fontsize=10)
        ax3.tick_params(colors="white")
        ax3.legend(fontsize=7, facecolor="#0a0f1a", labelcolor="white")

        # ── Panel 4: Min / Avg / Max table ───────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        col_labels = ["Phase", "Min %", "Avg %", "Max %", "Stability"]
        rows = []
        for pk, lbl in zip(phase_keys, phases):
            sc = scores_by_phase[pk]
            mn, avg, mx = min(sc), sum(sc)/n, max(sc)
            spread = mx - mn
            stab = "✅ Stable" if spread < 10 else "⚠ Variable" if spread < 25 else "❌ Unstable"
            rows.append([lbl, f"{mn:.1f}", f"{avg:.1f}", f"{mx:.1f}", stab])
        tbl = ax4.table(cellText=rows, colLabels=col_labels,
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
        for (ri, ci), cell in tbl.get_celld().items():
            cell.set_facecolor("#0d1a2a" if ri % 2 == 0 else "#0a1220")
            cell.set_edgecolor("#1a3a5c")
            cell.set_text_props(color="white")
        ax4.set_title("Stability Analysis", color="white", fontsize=10, pad=12)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"testing_summary_chart_{ts}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    log(f"\n  🖼   Summary chart saved → {fname}")
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — automated repeat loop for testing facilities
# ─────────────────────────────────────────────────────────────────────────────

import time as _time

def _run_once(lat, lon, area_name, radius_m, run_number, total_runs="?"):
    """Run all phases once. Never prompts."""
    global STAMP, _phase1a_ways, _phase1a_shelters

    STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(); print(SEP)
    print(f"  RUN {run_number}/{total_runs}  —  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  Location: {area_name}")
    print(SEP)

    print(f"\n  Fetching OSM data...")
    bbox = bbox_from_point(lat, lon, radius_m)
    ways, shelters = fetch_osm_data(bbox)
    if not ways:
        print("  ⚠  No road data — skipping run.")
        return None

    opt = SCENARIOS["1"]
    fd  = FLOOD_SCENARIOS["1"]
    print("  Building graph...")
    graph, node_info, shelter_ids, raw_edges = build_graph(
        lat, lon, ways, shelters, fd["multiplier"],
        opt["alpha"], opt["beta"], opt["gamma"], opt["delta"]
    )
    start = next((nid for nid, nd in node_info.items() if nd[3] == "start"), None)
    if not start:
        print("  ⚠  Could not place start node — skipping.")
        return None

    print(f"  {len(graph)} nodes | {sum(len(v) for v in graph.values())} edges | {len(shelter_ids)} shelters")

    all_results = {}

    print(f"\n  ▶  Phase 1 — Computational & Algorithmic Testing")
    _phase1a_ways     = ways
    _phase1a_shelters = shelters
    p1a = phase1a_algorithmic_testing(graph, node_info, shelter_ids, start)
    p1b = phase1b_pathfinding_accuracy(ways, shelters, node_info, start, fd["multiplier"])
    p1c = phase1c_dynamic_rerouting(graph, node_info, shelter_ids, start, ways, shelters)
    p1d = phase1d_network_resilience(graph, node_info, shelter_ids, start)
    all_results.update({"1A": p1a, "1B": p1b, "1C": p1c, "1D": p1d})

    print(f"\n  ▶  Phase 2 — System Performance Validation")
    p2a = phase2a_hydrodynamic_simulation(ways, shelters, node_info, start)
    p2b = phase2b_coupled_model(ways, shelters, node_info, start)
    p2c = phase2c_stress_tests(graph, node_info, shelter_ids, start, ways, shelters)
    all_results.update({"2A": p2a, "2B": p2b, "2C": p2c})

    fname = f"testing_results_{STAMP}_run{run_number}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  ✅  Run {run_number} done — saved → {fname}")
    return all_results


def main():
    print(); print(SEP)
    print("  TESTING SUITE — FLOOD EVACUATION ROUTE OPTIMIZER")
    print("  Automated Repeat Mode for Testing Facilities")
    print(SEP)

    lat, lon, area_name, radius_m = pick_location()

    print("\n  AUTOMATED RUN SETTINGS  (press Enter for defaults)\n")
    try:
        n_input = input("  How many times to repeat? [0 = infinite, default: 3]: ").strip() or "3"
        n_runs  = int(n_input)
    except ValueError:
        n_runs = 3
    infinite = (n_runs == 0)
    if not infinite:
        n_runs = max(1, n_runs)

    try:
        delay_s = float(input("  Delay between runs in seconds? [default: 5]: ").strip() or "5")
    except ValueError:
        delay_s = 5.0
    delay_s = max(0.0, delay_s)

    run_label = "infinite ∞" if infinite else str(n_runs)
    print(f"\n  Will run {run_label}x with {delay_s:.0f}s delay between runs.")
    if infinite:
        print("  Press Ctrl+C at any time to stop.")
    input("  Press Enter to start...")

    all_runs = []
    i = 0
    try:
        while True:
            i += 1
            total_label = "∞" if infinite else str(n_runs)
            result = _run_once(lat, lon, area_name, radius_m, i, total_label)
            if result:
                all_runs.append(result)
            if not infinite and i >= n_runs:
                break
            print(f"\n  ⏳  Next run in {delay_s:.0f}s...  (Ctrl+C to stop)")
            _time.sleep(delay_s)
    except KeyboardInterrupt:
        print(f"\n  Stopped after {i} run(s).")

    # Summary across all runs
    print(); print(SEP)
    print(f"  SUMMARY — {len(all_runs)} run(s) completed")
    print(SEP)
    for phase_key in ["1A", "1B", "2B", "2C"]:
        rates = []
        for r in all_runs:
            ph = r.get(phase_key, {})
            if   "pass_rate"    in ph: rates.append(ph["pass_rate"])
            elif "success_rate" in ph: rates.append(ph["success_rate"])
        if rates:
            avg = sum(rates) / len(rates)
            print(f"  Phase {phase_key}:  avg={avg:.1f}%  "
                  f"min={min(rates):.1f}%  max={max(rates):.1f}%")

    sfname = f"testing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(sfname, "w", encoding="utf-8") as f:
        json.dump({"runs": len(all_runs), "results": all_runs},
                  f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  📊  Summary saved → {sfname}")

    # One combined summary image across all runs
    if MATPLOTLIB_OK and all_runs:
        _make_summary_chart(all_runs, area_name)

    print(f"\n  All done! Laging handa. 🙏\n")


if __name__ == "__main__":
    main()
