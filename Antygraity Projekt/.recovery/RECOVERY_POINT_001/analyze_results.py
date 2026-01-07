
#!/usr/bin/env python3
"""
Analyze test results and generate report
"""

import json
import numpy as np
import os

def generate_dashboard():
    # Load results
    if not os.path.exists('test_results.json'):
        print("No results found. Run test_suite.py first.")
        return

    with open('test_results.json', 'r') as f:
        results = json.load(f)

    print("="*70)
    print("📊 ANTIGRAVITY SWARM - TEST ANALYSIS DASHBOARD")
    print("="*70)

    # Overall score
    pass_rate = results['pass_rate'] * 100
    print(f"\n🎯 OVERALL SCORE: {pass_rate:.1f}% ({results['passed']}/{results['total']})")

    # Category breakdown (Approximate grouping)
    categories = {
        'Learning': ['Test 1', 'Test 2', 'Test 5', 'Test 8'],
        'Evolution': ['Test 3', 'Test 9', 'Test 10'],
        'Intelligence': ['Test 4', 'Test 6'],
        'Stability': ['Test 7', 'Test 11', 'Test 12'],
    }

    print("\n📈 CATEGORY SCORES:")
    for cat, test_names in categories.items():
        # Filter tests that start with any of the names
        cat_tests = []
        for t in results['tests']:
            for name_start in test_names:
                if t['name'].startswith(name_start):
                    cat_tests.append(t)
                    break

        cat_pass = sum(1 for t in cat_tests if t['passed'])
        cat_total = len(cat_tests)
        cat_pct = (cat_pass / cat_total * 100) if cat_total > 0 else 0

        status = "✅" if cat_pct >= 80 else "⚠️" if cat_pct >= 60 else "❌"
        print(f"   {status} {cat}: {cat_pct:.0f}% ({cat_pass}/{cat_total})")


    # Key metrics
    print("\n🔬 KEY METRICS EXTRACTED:")
    for test in results['tests']:
        if test['metrics']:
            # Shorten name
            short_name = test['name'].split(':')[0]
            print(f"\n   {short_name}:")
            for metric, value in test['metrics'].items():
                print(f"      • {metric}: {value}")

    # Verdict
    print("\n" + "="*70)
    if pass_rate >= 90:
        print("🏆 VERDICT: EXCEPTIONAL - System demonstrates true learning")
    elif pass_rate >= 70:
        print("✅ VERDICT: STRONG - Clear evidence of genuine intelligence")
    elif pass_rate >= 50:
        print("⚠️ VERDICT: MODERATE - Some learning mechanisms working")
    else:
        print("❌ VERDICT: NEEDS WORK - Significant issues detected")
    print("="*70)

if __name__ == "__main__":
    generate_dashboard()
