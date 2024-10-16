"""
This class is responsible for sampling DFA formulas typically from
given template(s).

@ propositions: The set of propositions to be used in the sampled
                formula at random.
"""

import random
import numpy as np
from functools import reduce
from dfa import DFA, dict2dfa
import operator as OP

class DFASampler():
    def __init__(self, propositions):
        self.propositions = list(propositions)

    def sample(self):
        candidate = self._sample()
        while self.reject(candidate):
            candidate = self._sample()
        return candidate

    def _sample(self):
        raise NotImplementedError

    def reject(self, dfa_goal):
        mono = reduce(OP.and_, map(lambda dfa_clause: reduce(OP.or_, dfa_clause), dfa_goal))
        return mono.find_word() is None

class JoinSampler(DFASampler):
    def __init__(self, propositions, sampler_ids):
        super().__init__(propositions)
        self.n = len(sampler_ids)
        self.samplers = [getDFASampler(sampler_id, self.propositions) for sampler_id in sampler_ids]

    def _sample(self):
        return random.choice(self.samplers).sample()

class UntilTaskSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        assert 2*int(max_levels)*int(max_conjunctions) <= len(propositions), "The domain does not have enough propositions!"

    def _sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        p = random.sample(self.propositions,2*self.levels[1]*n_conjs)
        ltl = None
        seqs = []
        b = 0
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            # Sampling an until task of *n_levels* levels
            until_task = ('until',('not',p[b]),p[b+1])
            seq = [(p[b], p[b+1])]
            b +=2
            for j in range(1,n_levels):
                until_task = ('until',('not',p[b]),('and', p[b+1], until_task))
                seq = [(p[b], p[b+1])] + seq
                b +=2
            # Adding the until task to the conjunction of formulas that the agent have to solve
            if ltl is None: ltl = until_task
            else:           ltl = ('and',until_task,ltl)
            seqs = [tuple(seq)] + seqs
        seqs = tuple(seqs)
        def delta(s, c):
            if s is not None:
                for i in range(len(s)):
                    if s[i] != () and c != s[i][0][0] and c == s[i][0][1]:
                        return s[:i] + (s[i][1:],) + s[i + 1:]
                    elif s[i] != () and c == s[i][0][0]:
                        return None
            return s
        return ((DFA(
            start=seqs,
            inputs=self.propositions,
            label=lambda s: s == tuple(tuple() for _ in range(n_conjs)),
            transition=delta,
        ),),)

class CompositionalUntilTaskSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        assert 2*int(max_levels)*int(max_conjunctions) <= len(propositions), "The domain does not have enough propositions!"

    def _sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        p = random.sample(self.propositions,2*self.levels[1]*n_conjs)
        ltl = None
        seqs = []
        b = 0
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            # Sampling an until task of *n_levels* levels
            until_task = ('until',('not',p[b]),p[b+1])
            seq = [(p[b], p[b+1])]
            b +=2
            for j in range(1,n_levels):
                until_task = ('until',('not',p[b]),('and', p[b+1], until_task))
                seq = [(p[b], p[b+1])] + seq
                b +=2
            # Adding the until task to the conjunction of formulas that the agent have to solve
            if ltl is None: ltl = until_task
            else:           ltl = ('and',until_task,ltl)
            seqs = [tuple(seq)] + seqs
        seqs = tuple(seqs)
        def delta(s, c):
            if s is not None:
                if s != () and c != s[0][0] and c == s[0][1]:
                    return s[1:]
                elif s != () and c == s[0][0]:
                    return None
            return s
        dfas = tuple(DFA(start=seq, inputs=self.propositions, label=lambda s: s == tuple(), transition=delta) for seq in seqs)
        return tuple((dfa,) for dfa in dfas)

# This generates several sequence tasks which can be accomplished in parallel. 
# e.g. in (eventually (a and eventually c)) and (eventually b)
# the two sequence tasks are "a->c" and "b".
class EventuallySampler(DFASampler):
    def __init__(self, propositions, min_levels = 1, max_levels=4, min_conjunctions=1, max_conjunctions=3):
        super().__init__(propositions)
        assert(len(propositions) >= 3)
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        self.levels = (int(min_levels), int(max_levels))

    def _sample(self):
        conjs = random.randint(*self.conjunctions)
        seqs = tuple(self.sample_sequence() for _ in range(conjs))
        def delta(s, c):
            for i in range(len(s)):
                if s[i] != () and c in s[i][0]:
                    return s[:i] + (s[i][1:],) + s[i + 1:]
            return s
        return ((DFA(
            start=seqs,
            inputs=self.propositions,
            label=lambda s: s == tuple(tuple() for _ in range(conjs)),
            transition=delta,
        ),),)

    def sample_sequence(self):
        length = random.randint(*self.levels)
        seq = []

        last = []
        while len(seq) < length:
            # Randomly replace some propositions with a disjunction to make more complex formulas
            population = [p for p in self.propositions if p not in last]

            if random.random() < 0.25:
                c = random.sample(population, 2)
            else:
                c = random.sample(population, 1)

            seq.append(tuple(c))
            last = c

        return tuple(seq)

class CompositionalEventuallySampler(EventuallySampler):
    def __init__(self, propositions, min_levels = 1, max_levels=4, min_conjunctions=1, max_conjunctions=3):
        super().__init__(propositions, min_levels, max_levels, min_conjunctions, max_conjunctions)
        assert(len(propositions) >= 3)
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        self.levels = (int(min_levels), int(max_levels))

    def _sample(self):
        conjs = random.randint(*self.conjunctions)
        seqs = tuple(self.sample_sequence() for _ in range(conjs))
        dfas = tuple(DFA(start=seq, inputs=self.propositions, label=lambda s: s == tuple(), transition=lambda s, c: s[1:] if s != () and c in s[0] else s) for seq in seqs)
        return tuple((dfa,) for dfa in dfas)

class AdversarialEnvSampler(DFASampler):
    def _sample(self):
        p = random.randint(0,1)
        if p == 0:
            def delta(s, c):
                if s == 0 and c == 'a':
                    return 1
                elif s == 1 and c == 'b':
                    return 2
                return s
            return ((DFA(
                start=0,
                inputs=self.propositions,
                label=lambda s: s == 2,
                transition=delta,
            ),),)
        else:
            def delta(s, c):
                if s == 0 and c == 'a':
                    return 1
                elif s == 1 and c == 'c':
                    return 2
                return s
            return ((DFA(
                start=0,
                inputs=self.propositions,
                label=lambda s: s == 2,
                transition=delta,
            ),),)

class ReachSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))

    def _sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        seqs = []
        p = self.propositions
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            seq = []
            for j in range(n_levels):
                random.shuffle(p)
                seq.append((p[0],))
            seqs.append(tuple(seq))
        seqs = tuple(seqs)
        def delta(s, c):
            if s is not None:
                for i in range(len(s)):
                    if s[i] != () and c == s[i][0][0]:
                        return s[:i] + (s[i][1:],) + s[i + 1:]
            return s
        return ((DFA(
            start=seqs,
            inputs=self.propositions,
            label=lambda s: s == tuple(tuple() for _ in range(n_conjs)),
            transition=delta,
        ).minimize(),),)

class CompositionalReachSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))

    def _sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        seqs = []
        p = self.propositions
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            seq = []
            for j in range(n_levels):
                random.shuffle(p)
                seq.append((p[0],))
            seqs.append(tuple(seq))
        seqs = tuple(seqs)
        def delta(s, c):
            if s is not None:
                if s != () and c == s[0][0]:
                    return s[1:]
            return s
        dfas = tuple(DFA(start=seq, inputs=self.propositions, label=lambda s: s == tuple(), transition=delta).minimize() for seq in seqs)
        return tuple((dfa,) for dfa in dfas)

class ReachAvoidSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))

    def _sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        seqs = []
        p = self.propositions
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            seq = []
            for j in range(n_levels):
                random.shuffle(p)
                seq.append((p[0], p[1]))
            seqs.append(tuple(seq))
        seqs = tuple(seqs)
        def delta(s, c):
            if s is not None:
                for i in range(len(s)):
                    if s[i] != () and c != s[i][0][0] and c == s[i][0][1]:
                        return s[:i] + (s[i][1:],) + s[i + 1:]
                    elif s[i] != () and c == s[i][0][0]:
                        return None
            return s
        return ((DFA(
            start=seqs,
            inputs=self.propositions,
            label=lambda s: s == tuple(tuple() for _ in range(n_conjs)),
            transition=delta,
        ).minimize(),),)

class CompositionalReachAvoidSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))

    def _sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        seqs = []
        p = self.propositions
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            seq = []
            for j in range(n_levels):
                random.shuffle(p)
                seq.append((p[0], p[1]))
            seqs.append(tuple(seq))
        seqs = tuple(seqs)
        def delta(s, c):
            if s is not None:
                if s != () and c != s[0][0] and c == s[0][1]:
                    return s[1:]
                elif s != () and c == s[0][0]:
                    return None
            return s
        dfas = tuple(DFA(start=seq, inputs=self.propositions, label=lambda s: s == tuple(), transition=delta).minimize() for seq in seqs)
        return tuple((dfa,) for dfa in dfas)

class ReachAvoidRedemptionSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))

    def _sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        seqs = []
        p = self.propositions
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            seq = []
            for j in range(n_levels):
                random.shuffle(p)
                seq.append((p[0], p[1], p[2]))
            seqs.append(tuple(seq))
        seqs = tuple(seqs)
        def delta(s, c):
            s, is_in_recovery_mode = s
            if is_in_recovery_mode:
                for i in range(len(s)):
                    if s[i] != () and c == s[i][0][2]: # Fix
                        return s, False
            else:
                for i in range(len(s)):
                    if s[i] != () and c == s[i][0][0]: # Reach
                        return s[:i] + (s[i][1:],) + s[i + 1:], False
                    elif s[i] != () and c == s[i][0][1]: # Avoid
                        return s, True
            return s, is_in_recovery_mode
        return ((DFA(
            start=(seqs, False),
            inputs=self.propositions,
            label=lambda s: s[0] == tuple(tuple() for _ in range(n_conjs)) and not s[1],
            transition=delta,
        ).minimize(),),)

class CompositionalReachAvoidRedemptionSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))

    def _sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        seqs = []
        p = self.propositions
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            seq = []
            for j in range(n_levels):
                random.shuffle(p)
                seq.append((p[0], p[1], p[2]))
            seqs.append(tuple(seq))
        seqs = tuple(seqs)
        def delta(s, c):
            is_in_recovery_mode, s = s
            if is_in_recovery_mode:
                if s != () and c == s[0][2]: # Fix
                    return False, s
            else:
                if s != () and c == s[0][0]: # Reach
                    return False, s[1:]
                elif s != () and c == s[0][1]: # Avoid
                    return True, s
            return is_in_recovery_mode, s
        dfas = tuple(DFA(start=(False, seq), inputs=self.propositions, label=lambda s: not s[0] and s[1] == tuple(), transition=delta).minimize() for seq in seqs)
        return tuple((dfa,) for dfa in dfas)

class ParitySampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))

    def _sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        seqs = []
        p = self.propositions
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            seq = []
            for j in range(n_levels):
                random.shuffle(p)
                seq.append((p[0], p[1]))
            seqs.append(tuple(seq))
        seqs = tuple(seqs)
        def delta(s, c):
            s, is_in_recovery_mode = s
            if is_in_recovery_mode:
                for i in range(len(s)):
                    if s[i] != () and c == s[i][0][1]: # Even
                        return s, False
            else:
                for i in range(len(s)):
                    if s[i] != () and c == s[i][0][0]: # Reach
                        return s[:i] + (s[i][1:],) + s[i + 1:], False
                    elif s[i] != () and c == s[i][0][1]: # Odd
                        return s, True
            return s, is_in_recovery_mode
        return ((DFA(
            start=(seqs, False),
            inputs=self.propositions,
            label=lambda s: s[0] == tuple(tuple() for _ in range(n_conjs)) and not s[1],
            transition=delta,
        ).minimize(),),)

class CompositionalParitySampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))

    def _sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        seqs = []
        p = self.propositions
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            seq = []
            for j in range(n_levels):
                random.shuffle(p)
                seq.append((p[0], p[1]))
            seqs.append(tuple(seq))
        seqs = tuple(seqs)
        def delta(s, c):
            is_in_recovery_mode, s = s
            if is_in_recovery_mode:
                if s != () and c == s[0][1]: # Even
                    return False, s
            else:
                if s != () and c == s[0][0]: # Reach
                    return False, s[1:]
                elif s != () and c == s[0][1]: # Odd
                    return True, s
            return is_in_recovery_mode, s
        dfas = tuple(DFA(start=(False, seq), inputs=self.propositions, label=lambda s: not s[0] and s[1] == tuple(), transition=delta).minimize() for seq in seqs)
        return tuple((dfa,) for dfa in dfas)

class ReachAvoidDerivedSampler(DFASampler):
    def __init__(self, propositions, truncate=True):
        super().__init__(propositions)
        self.truncate = truncate
        self.sampler = self.dfa_sampler()
        self.p = 0.5
        self.max_size = 8 # + 2 = 10 states in total
        self.n_values = np.array(list(range(1, self.max_size + 1)))
        self.n_p = np.array([(self.p)**v for v in self.n_values])
        self.n_p = self.n_p / np.sum(self.n_p)

    def reach_avoid_sampler(self, prob_stutter=0.9):
        n_tokens = len(self.propositions)
        assert n_tokens > 1

        if self.truncate:
            n = 2 + np.random.choice(self.n_values, p=self.n_p)
        else:
            n = 2 + np.random.geometric(p=self.p)
        success, fail = n - 2, n - 1

        tokens = list(self.propositions)
        while True:
            transitions = {
              success: (True,  {t: success for t in tokens}),
              fail:    (False, {t: fail    for t in tokens}),
            }
            for state in range(n - 2):
                noop, good, bad = partition = (set(), set(), set())
                random.shuffle(tokens)
                good.add(tokens[0])
                bad.add(tokens[1])
                for token in tokens[2:]:
                    if random.random() <= prob_stutter:
                        noop.add(token)
                    else:
                        partition[random.randint(1, 2)].add(token)

                _transitions = dict()
                for token in good:
                    _transitions[token] = state + 1
                for token in bad:
                    _transitions[token] = fail
                for token in noop:
                    _transitions[token] = state

                transitions[state] = (False, _transitions)

            yield dict2dfa(transitions, start=0).minimize()


    def accepting_is_sink(self, d: DFA):
        def transition(s, c):
            if d._label(s) is True:
                return s
            return d._transition(s, c)
        return DFA(start=d.start,
                   inputs=d.inputs,
                   label=d._label,
                   transition=transition)


    def dfa_sampler(self, max_mutations=5):
        dfas = self.reach_avoid_sampler()
        while True:
            candidate = next(dfas)
            for _ in range(random.randint(0, max_mutations)):
                tmp =  self.accepting_is_sink(self.change_transition(candidate))
                if tmp is None: continue
                tmp = tmp.minimize()
                if len(tmp.states()) == 1: continue
                candidate = tmp.minimize()
            yield candidate

    def _sample(self):
        sample = next(self.sampler)
        return ((sample,),)

    def change_transition(self, orig: DFA, rng=random):
        if (len(orig.inputs) <= 1) or (len(orig.states()) <= 1):
            return None

        state1 = rng.choice(list(orig.states()))
        state2 = rng.choice(list(orig.states()))
        sym = rng.choice(list(orig.inputs))

        def transition(s, c):
            if (s, c) == (state1, sym):
                return state2
            return orig._transition(s, c)

        return DFA(
            label=orig._label, transition=transition,
            start=orig.start, inputs=orig.inputs, outputs=orig.outputs,
        )

class CompositionalReachAvoidDerivedSampler(DFASampler):
    def __init__(self, propositions, truncate=True):
        super().__init__(propositions)
        self.truncate = truncate
        self.sampler = ReachAvoidDerivedSampler(self.propositions, truncate=truncate).sampler
        self.p = 0.5
        self.max_conjs = 5
        self.n_conjs_values = np.array(list(range(1, self.max_conjs + 1)))
        self.n_conjs_p = np.array([(self.p)**v for v in self.n_conjs_values])
        self.n_conjs_p = self.n_conjs_p / np.sum(self.n_conjs_p)

    def _sample(self):
        if self.truncate:
            n_conjs = np.random.choice(self.n_conjs_values, p=self.n_conjs_p)
        else:
            n_conjs = np.random.geometric(p=self.p)
        dfas = tuple(next(self.sampler) for _ in range(n_conjs))
        return tuple((dfa,) for dfa in dfas)

class BroadcastNegation(DFASampler):
    def __init__(self, dfa_sampler):
        super().__init__(dfa_sampler.propositions)
        self.dfa_sampler = dfa_sampler

    def _sample(self):
        dfa_goal = self.dfa_sampler.sample()
        return self._negate(dfa_goal)

    @classmethod
    def _negate(self, dfa_goal):
        return tuple(tuple((~dfa).minimize() for dfa in dfa_clause) for dfa_clause in dfa_goal)

class RandomBroadcastNegation(DFASampler):
    def __init__(self, dfa_sampler):
        super().__init__(dfa_sampler.propositions)
        self.dfa_sampler = dfa_sampler

    def _sample(self):
        dfa_goal = self.dfa_sampler.sample()
        return self._negate(dfa_goal)

    @classmethod
    def _negate(self, dfa_goal):
        broadcast_negated_dfa_goal = []
        is_first = True
        for dfa_clause in dfa_goal:
            broadcast_negated_dfa_clause = []
            for dfa in dfa_clause:
                if is_first:
                    broadcast_negated_dfa_clause.append(dfa)
                    is_first = False
                else:
                    if random.random() < 0.5:
                        broadcast_negated_dfa_clause.append((~dfa).minimize())
                    else:
                        broadcast_negated_dfa_clause.append(dfa)
            broadcast_negated_dfa_goal.append(tuple(broadcast_negated_dfa_clause))
        broadcast_negated_dfa_goal = tuple(broadcast_negated_dfa_goal)
        return broadcast_negated_dfa_goal


def getRegisteredSamplers(propositions):
    raise NotImplementedError

# The DFASampler factory method that instantiates the proper sampler
# based on the @sampler_id.
def getDFASampler(sampler_id, propositions):
    tokens = ["Default"]
    if (sampler_id != None):
        tokens = sampler_id.split("_")

    # Don't change the order of ifs here otherwise the OR sampler will fail
    if ("_JOIN_" in sampler_id): # e.g., Eventually_1_5_1_4_JOIN_Until_1_3_1_2
        sampler_ids = sampler_id.split("_JOIN_")
        return JoinSampler(propositions, sampler_ids)
    elif (tokens[0] == "ReachAvoid"):
        return ReachAvoidSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "N-ReachAvoid"):
        return BroadcastNegation(ReachAvoidSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "CompositionalReachAvoid"):
        return CompositionalReachAvoidSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "BN-CompositionalReachAvoid"):
        return BroadcastNegation(CompositionalReachAvoidSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "RBN-CompositionalReachAvoid"):
        return RandomBroadcastNegation(CompositionalReachAvoidSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "Reach"):
        return ReachSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "N-Reach"):
        return BroadcastNegation(ReachSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "CompositionalReach"):
        return CompositionalReachSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "BN-CompositionalReach"):
        return BroadcastNegation(CompositionalReachSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "RBN-CompositionalReach"):
        return RandomBroadcastNegation(CompositionalReachSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "ReachAvoidRedemption"):
        return ReachAvoidRedemptionSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "N-ReachAvoidRedemption"):
        return BroadcastNegation(ReachAvoidRedemptionSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "CompositionalReachAvoidRedemption"):
        return CompositionalReachAvoidRedemptionSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "BN-CompositionalReachAvoidRedemption"):
        return BroadcastNegation(CompositionalReachAvoidRedemptionSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "RBN-CompositionalReachAvoidRedemption"):
        return RandomBroadcastNegation(CompositionalReachAvoidRedemptionSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "Parity"):
        return ParitySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "N-Parity"):
        return BroadcastNegation(ParitySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "CompositionalParity"):
        return CompositionalParitySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "BN-CompositionalParity"):
        return BroadcastNegation(CompositionalParitySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "RBN-CompositionalParity"):
        return RandomBroadcastNegation(CompositionalParitySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "Until"):
        return UntilTaskSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "N-Until"):
        return BroadcastNegation(UntilTaskSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "CompositionalUntil"):
        return CompositionalUntilTaskSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "BN-CompositionalUntil"):
        return BroadcastNegation(CompositionalUntilTaskSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "RBN-CompositionalUntil"):
        return RandomBroadcastNegation(CompositionalUntilTaskSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "Adversarial"):
        return AdversarialEnvSampler(propositions)
    elif (tokens[0] == "N-Adversarial"):
        return BroadcastNegation(AdversarialEnvSampler(propositions))
    elif (tokens[0] == "Eventually"):
        return EventuallySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "N-Eventually"):
        return BroadcastNegation(EventuallySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "CompositionalEventually"):
        return CompositionalEventuallySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "BN-CompositionalEventually"):
        return BroadcastNegation(CompositionalEventuallySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "RBN-CompositionalEventually"):
        return RandomBroadcastNegation(CompositionalEventuallySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4]))
    elif (tokens[0] == "ReachAvoidDerived"):
        return ReachAvoidDerivedSampler(propositions, truncate=True)
    elif (tokens[0] == "N-ReachAvoidDerived"):
        return BroadcastNegation(ReachAvoidDerivedSampler(propositions, truncate=True))
    elif (tokens[0] == "CompositionalReachAvoidDerived"):
        return CompositionalReachAvoidDerivedSampler(propositions, truncate=True)
    elif (tokens[0] == "BN-CompositionalReachAvoidDerived"):
        return BroadcastNegation(CompositionalReachAvoidDerivedSampler(propositions, truncate=True))
    elif (tokens[0] == "RBN-CompositionalReachAvoidDerived"):
        return RandomBroadcastNegation(CompositionalReachAvoidDerivedSampler(propositions, truncate=True))
    elif (tokens[0] == "NT-ReachAvoidDerived"):
        return ReachAvoidDerivedSampler(propositions, truncate=False)
    elif (tokens[0] == "N-NT-ReachAvoidDerived"):
        return BroadcastNegation(ReachAvoidDerivedSampler(propositions, truncate=False))
    elif (tokens[0] == "NT-CompositionalReachAvoidDerived"):
        return CompositionalReachAvoidDerivedSampler(propositions, truncate=False)
    elif (tokens[0] == "BN-NT-CompositionalReachAvoidDerived"):
        return BroadcastNegation(CompositionalReachAvoidDerivedSampler(propositions, truncate=False))
    elif (tokens[0] == "RBN-NT-CompositionalReachAvoidDerived"):
        return RandomBroadcastNegation(CompositionalReachAvoidDerivedSampler(propositions, truncate=False))
    else:
        raise NotImplementedError

