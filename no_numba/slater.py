from util import *

class SlaterCondon:
    def __init__(self, configs, h1e, h2e):
        self.configs = configs
        self.h1e = h1e
        self.h2e = h2e

    def e1_kk(self, det_label: int) -> float:
        ''' Slater-Condon energy value calculator for the one-electron repulsion,
            no orbital change case

            Energy = Sum <m|h|m> (= [m|h|m]) foreach m
    
        Args:
            det_label (int): The label of the determinent being computed
        
        Returns:
            float: Calculated energy value
        '''

        e1 = 0.
        for i in self.configs[det_label, :]:
            sm = spatial(i)
            e1 += self.h1e[sm, sm]
        return e1


    def e2_kk(self, det_label: int) -> float:
        ''' Slater-Condon energy value calculator for the two-electron repulsion,
            no orbital change case
            
            Energy = 0.5 * Sum <mn||pn> (= [mp|nn] - [mn|np]) foreach m foreach n
        
        Args:
            det_label (int): The label of the determinent being computed
            
        Returns:
            float: Calculated energy value
        '''
        e2 = 0.
        for sm in self.configs[det_label, :]:
            for sn in self.configs[det_label]:
                m, n = spatial(sm), spatial(sn)
                coulomb = self.h2e[m, m, n, n]
                exchange = self.h2e[m, n, n, m] if spin(sm) == spin(sn) else 0.0
                e2 += coulomb - exchange
        return 0.5 * e2

    def e1_kl1(self, m: int, n: int) -> float:
        ''' Slater-Condon energy value calculator for the one-electron repulsion,
            with one orbital difference
            
            Energy = Sum <m|h|p> (= [m|h|p])
            Where m and p are equivalent spins
        
        Args:
            m (int): The electron label in the first configuration that is different
            p (int): The electron label in the second configuration that is different
            
        Returns:
            float: Calculated energy value
        '''
        return self.h1e[spatial(m), spatial(n)] if spin(m) == spin(n) else 0.


    def e2_kl1(self, det_label: int, m: int, p: int) -> float:
        ''' Slater-Condon energy value calculator for the two-electron repulsion,
            with one orbital difference
                
            Energy = <mn||pn> (= [mp|nn] - [mn|np]) foreach n
            Where m and p, m and n are equivalent spins
            
        Args:
            m (int): The electron label in the first configuration that is different
            p (int): The electron label in the second configuration that is different
                
        Returns:
            float: Calculated energy value
        '''
    
        if spin(m) != spin(p): return 0.0
    
        e2 = 0.
    
        for n in self.configs[det_label, :]:  # Sums over all n
            sm, sn, sp = spatial(m), spatial(n), spatial(p)
            coulomb = self.h2e[sm, sp, sn, sn]
            exchange = self.h2e[sm, sn, sn, sp] if spin(m) == spin(n) else 0.0
            e2 += coulomb - exchange
        return e2


    def e2_kl2(self, m: int, n: int, p: int, q: int) -> float:
        ''' Slater-Condon energy value calculator for the two-electron repulsion,
            with two orbital differences
                
            Energy = <mn||pq> (= [mp|nq] - [mq|np])
            Where m and p, m and n are equivalent spins
            
        Args:
            m (int): The electron label in the first configuration that is different
            p (int): The electron label in the second configuration that is different
                
        Returns:
            float: Calculated energy value
        '''
        sm, sn, sp, sq = spatial(m), spatial(n), spatial(p), spatial(q)
        coulomb = self.h2e[sm, sp, sn, sq] if spin(m) == spin(p) and spin(n) == spin(q) else 0.0
        exchange = self.h2e[sm, sq, sn, sp] if spin(m) == spin(q) and spin(n) == spin(p) else 0.0
        return coulomb - exchange