import numpy
from petsc4py import PETSc
import dolfin
import pdb


class Restriction():

    def __init__(self, function_spaces, local_rdofs):
        self.function_space = function_spaces
        self.local_rdofs = local_rdofs

        self.global_rdofs = []

        # Compute offsets in the non-restricted matrix
        self.offsets = [0]
        offset = 0
        for space in function_spaces:
            self.offsets.append(self.offsets[-1] + space.dofmap.index_map.size_local +
                                space.dofmap.index_map.num_ghosts)
            offset += self.offsets[-1]

        # Shift the block-local dof indices by offsets
        for i, rdofsi in enumerate(self.local_rdofs):
            dofs = rdofsi.copy()
            dofs += self.offsets[i]
            self.global_rdofs.append(dofs)

        rdofs = numpy.hstack(self.global_rdofs)
        self.isrow = PETSc.IS().createBlock(1, rdofs)

    def restrict_matrix(self, A):
        subA = A.createSubMatrix(isrow=self.isrow, iscol=self.isrow)
        subA.assemble()

        return subA

    def restrict_vector(self, x):
        subx = x.getSubVector(self.isrow)

        return subx

    def update_functions(self, f, rx):
        rdof_offset = 0
        for i, fi in enumerate(f):
            num_rdofs = self.global_rdofs[i].shape[0]
            fi.vector.array[self.global_rdofs[i] - self.offsets[i]] = rx.array_r[rdof_offset:(rdof_offset + num_rdofs)]
            rdof_offset += num_rdofs

    def extend_vector(self, rx, x):
        for i, rdofs in enumerate(self.global_rdofs):
            num_rdofs = rdofs.shape[0]
            x.array[rdofs - self.offsets[i]] = rx.array_r[rdof_offset:(rdof_offset + num_rdofs)]
            rdof_offset += num_rdofs
