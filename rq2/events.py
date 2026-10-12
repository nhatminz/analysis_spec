"""Adapt packed SPARK trees to the existing, unchanged VerificationLog schema."""
from types import SimpleNamespace
from e2e6.events import VerificationLog


class PackedVerificationLog(VerificationLog):
    def packed_verification(self, *, tree, proposals, sampled, path, **metadata):
        parents=tree.parents.detach().cpu().tolist()
        tokens=tree.tokens.detach().cpu().tolist()
        contexts=tree.feedback_contexts.detach().cpu().tolist()
        samples=sampled.detach().cpu().tolist()
        lengths=path.lengths.detach().cpu().tolist()
        emitted=path.tokens.detach().cpu().tolist()
        packed_paths=path.packed_indices.detach().cpu().tolist()
        candidates=[{} for _ in parents]
        for offset,ids in proposals:
            for row,values in enumerate(ids.detach().cpu().tolist()):
                for context,values_at_context in enumerate(values):candidates[row][offset+context]=values_at_context
        trees=[];chosen=[];observations=[];paths=[];accepted=[]
        k=metadata['draft_k'];past=metadata['past_kv_len']
        for row,par in enumerate(parents):
            if par[0]!=-1:raise ValueError('Packed verification root has a parent')
            roots=[j for j in range(1,len(par)) if par[j]==0]
            others=[j for j in range(1,len(par)) if par[j]!=0]
            if len(roots)>k:raise ValueError('Too many retained root children')
            # VerificationLog identifies root children by their original ID < k.
            # Non-root packed IDs must therefore be remapped above k, even if
            # pruning kept fewer than k root children. Holes are never visited.
            mapping={j:i for i,j in enumerate(roots)}
            mapping.update({j:k+i for i,j in enumerate(others)})
            nodes=[None]*(k+len(others));depths={0:0}
            for j in range(1,len(par)):
                parent=par[j]
                if parent<0 or parent>=j:raise ValueError('Packed tree is not parent-closed')
                depths[j]=depths[parent]+1
                nodes[mapping[j]]=SimpleNamespace(input_id=tokens[row][j],depth=depths[j]-1,
                                                 child=[],target_index=past+j)
                if parent:nodes[mapping[parent]].child.append(mapping[j])
            observation={}
            for j,context in enumerate(contexts[row]):
                if context>=0:
                    if context not in candidates[row]:raise ValueError('Computed proposal context missing from observation')
                    observation[-1 if j==0 else mapping[j]]=candidates[row][context]
            n=int(lengths[row])
            if not 1<=n<=len(emitted[row]):raise ValueError('Empty/invalid accepted verification path')
            trees.append(nodes);chosen.append([mapping[j] for j in range(1,len(par))])
            observations.append(observation)
            paths.append([past+j for j in packed_paths[row][:n]])
            accepted.append(emitted[row][:n])
        self.verification(trees=trees,chosen=chosen,proposals=observations,sampled=samples,
                          paths=paths,emitted=accepted,**metadata)
