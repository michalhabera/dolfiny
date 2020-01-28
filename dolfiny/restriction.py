import typing

import numpy
import dolfinx
from petsc4py import PETSc


class Restriction():
    def __init__(self, function_spaces: typing.List[dolfinx.FunctionSpace], blocal_dofs: typing.List[numpy.ndarray]):
        """Restriction of a problem to subset of degree-of-freedom indices.

        Parameters
        ----------
        function_spaces
        blocal_dofs
            Block-local DOF indices.

        Note
        ----
        Currently, restriction of a matrix and vector is sub-optimal, since it assumes
        different parallel layout every time restriction is called.

        """
        self.function_space = function_spaces
        self.blocal_dofs = blocal_dofs

        self.comm = dolfinx.MPI.comm_world

        self.bglobal_dofs = []
        self.bglobal_dofs_ng = []

        self.boffsets = [0]
        self.boffsets_ng = [0]
        offset = 0
        offset_ng = 0

        for i, space in enumerate(function_spaces):
            # Compute block-offsets in the non-restricted matrix
            # _ng version for offsets with ghosts excluded
            self.boffsets.append(self.boffsets[-1] + space.dofmap.index_map.size_local
                                 + space.dofmap.index_map.num_ghosts)
            self.boffsets_ng.append(self.boffsets_ng[-1] + space.dofmap.index_map.size_local)
            offset += self.boffsets[-1]
            offset_ng += self.boffsets_ng[-1]

            # Compute block-global dof indices in non-restricted matrix
            # _ng version for offsets with ghosts excluded
            dofs = self.blocal_dofs[i].copy()
            dofs += self.boffsets[i]
            self.bglobal_dofs.append(dofs)

            dofs_ng = self.blocal_dofs[i].copy()
            dofs_ng += self.boffsets_ng[i]
            self.bglobal_dofs_ng.append(dofs_ng)

        self.bglobal_dof = numpy.hstack(self.bglobal_dofs)
        self.bglobal_dof_ng = numpy.hstack(self.bglobal_dofs_ng)

    def restrict_matrix(self, A: PETSc.Mat):
        global_isrow = PETSc.IS(self.comm).createGeneral(self.bglobal_dof)
        global_isrow = A.getLGMap()[0].applyIS(global_isrow)

        subA = A.createSubMatrix(isrow=global_isrow, iscol=global_isrow)
        subA.assemble()

        return subA

    def restrict_vector(self, x: PETSc.Vec):
        global_isrow = PETSc.IS(self.comm).createGeneral(self.bglobal_dof_ng)
        global_isrow = x.getLGMap().applyIS(global_isrow)

        subx = x.getSubVector(global_isrow)
        return subx

    def update_functions(self, f: typing.List, rx: PETSc.Vec):
        """Update dolfinx Functions using restricted DOF indices."""
        rdof_offset = 0
        for i, fi in enumerate(f):
            num_rdofs = self.bglobal_dofs[i].shape[0]
            fi.vector.array[self.bglobal_dofs[i] - self.boffsets[i]] = rx.array_r[rdof_offset:(rdof_offset + num_rdofs)]
            fi.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            rdof_offset += num_rdofs
