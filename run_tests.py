#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一测试入口。"""

import subprocess
import sys

TESTS = [
    'tests/test_eval_v2.py',
    'tests/test_shanten_ukeire.py',
    'tests/legacy_test.py',
    'tests/test_eval2_used_paircoef.py',
]


def main():
    failed = []
    for script in TESTS:
        print('\n' + '=' * 60)
        print('Running', script)
        print('=' * 60)
        rc = subprocess.call([sys.executable, script])
        if rc != 0:
            failed.append(script)

    print('\n' + '=' * 60)
    if failed:
        print('FAILED:', ', '.join(failed))
        return 1
    print('All tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
