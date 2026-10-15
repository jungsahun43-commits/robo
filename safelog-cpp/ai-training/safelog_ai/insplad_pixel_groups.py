"""Conservative connected groups and one fixed allocation of normal candidates."""
import hashlib
import math

DEVELOPMENT=frozenset(('fit','validation','cal'))


class Groups:
    def __init__(self,n):self.parents=list(range(n))
    def find(self,i):
        while self.parents[i]!=i:
            self.parents[i]=self.parents[self.parents[i]];i=self.parents[i]
        return i
    def union(self,a,b):
        a,b=self.find(a),self.find(b);self.parents[max(a,b)]=min(a,b)


def allocate(group_keys,held,prior_roles):
    result={};unlocked=[]
    for key in sorted(set(group_keys)):
        roles=set(prior_roles.get(key,()))
        if not roles<=DEVELOPMENT:raise ValueError('Unexpected development role')
        if key in held:result[key]='excluded_TEST_connected'
        elif len(roles)>1:result[key]='excluded_prior_role_conflict'
        elif roles:result[key]=next(iter(roles))
        else:unlocked.append(key)
    ordered=sorted(unlocked,key=lambda k:(hashlib.sha256(('insplad57:'+k).encode()).hexdigest(),k))
    n=math.ceil(.2*len(ordered))
    for i,key in enumerate(ordered):result[key]='validation' if i<n else 'cal' if i<2*n else 'fit'
    return result
