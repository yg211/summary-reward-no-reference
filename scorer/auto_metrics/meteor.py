#!/usr/bin/env python

# Python wrapper for METEOR implementation

import os
import pexpect

from resources import METEOR_DIR

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = os.environ.get('METEOR_JAR', os.path.join(METEOR_DIR,'meteor-1.5','meteor-1.5.jar'))

class Meteor:
    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                '-', '-', '-stdio', '-l', 'en', '-norm']
        self.child = pexpect.spawn(
            self.meteor_cmd[0], self.meteor_cmd[1:],
            cwd=os.path.dirname(os.path.abspath(__file__)), encoding="utf8")

    def _comm(self, line):
        line = line.strip()
        self.child.sendline(line)
        resp = self.child.readline().strip()
        assert resp == line, "Expected {}, got {}".format(line, resp)
        resp = self.child.readline().strip()
        return resp

    def score(self, hypothesis_str, reference_list):
        #self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace("\n", " ").replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str)).strip()

        stats = self._comm(score_line)

        eval_line = 'EVAL ||| {}'.format(stats).strip()
        # EVAL ||| stats
        score = float(self._comm(eval_line))
        return score

    def __del__(self):
        if self.child:
            self.child.close()


