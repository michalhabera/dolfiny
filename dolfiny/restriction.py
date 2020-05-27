import typing

import numpy
import dolfinx
from petsc4py import PETSc


class Restriction():
    def __init__(self, function_spaces: typing.List[dolfinx.FunctionSpace],
                 blocal_dofs: typing.List[numpy.ndarray], comm=None):
        """Restriction of a problem to subset of degree-of-freedom indices.

        Parameters
        ----------
        function_spaces
        blocal_dofs
            Block-local DOF indices.
        comm: optional

        Note
        ----
        Currently, restriction of a matrix and vector is sub-optimal, since it assumes
        different parallel layout every time restriction is called.

        """
        self.function_space = function_spaces
        self.blocal_dofs = blocal_dofs

        if comm is None:
            self.comm = self.function_space[0].mesh.mpi_comm
        else:
            self.comm = comm

        self.bglobal_dofs = []
        self.bglobal_dofs_ng = []

        self.boffsets = [0]
        self.boffsets_ng = [0]
        offset = 0
        offset_ng = 0

        for i, space in enumerate(function_spaces):

            bs = space.dofmap.index_map.block_size

            size_local = space.dofmap.index_map.size_local
            num_ghosts = space.dofmap.index_map.num_ghosts

            # Compute block-offsets in the non-restricted function space
            # _ng version for offsets with ghosts excluded
            self.boffsets.append(self.boffsets[-1] + bs * (size_local + num_ghosts))
            self.boffsets_ng.append(self.boffsets_ng[-1] + bs * size_local)
            offset += self.boffsets[-1]
            offset_ng += self.boffsets_ng[-1]

            # Compute block-global dof indices in non-restricted function space
            # _ng version for offsets with ghosts excluded
            dofs = self.blocal_dofs[i].copy()
            # Remove any ghost dofs
            dofs = dofs[dofs < bs * size_local]
            dofs += self.boffsets[i]
            self.bglobal_dofs.append(dofs)

            dofs_ng = self.blocal_dofs[i].copy()
            # Remove any ghost dofs
            dofs_ng = dofs_ng[dofs_ng < bs * size_local]
            dofs_ng += self.boffsets_ng[i]
            self.bglobal_dofs_ng.append(dofs_ng)

        self.bglobal_dof = numpy.hstack(self.bglobal_dofs)
        self.bglobal_dof_ng = numpy.hstack(self.bglobal_dofs_ng)

    def restrict_matrix(self, A: PETSc.Mat):
        local_isrow = PETSc.IS(self.comm).createBlock(1, self.bglobal_dof)
        global_isrow = A.getLGMap()[0].applyIS(local_isrow)

        subA = A.createSubMatrix(isrow=global_isrow, iscol=global_isrow)
        subA.assemble()

        return subA

    def restrict_vector(self, x: PETSc.Vec):
        local_isrow = PETSc.IS(self.comm).createBlock(1, self.bglobal_dof_ng)
        global_isrow = x.getLGMap().applyIS(local_isrow)

        subx = x.getSubVector(global_isrow)
        return subx

    def update_functions(self, f: typing.List, rx: PETSc.Vec):
        """Update Functions using restricted DOF indices."""
        rdof_offset = 0
        for i, fi in enumerate(f):
            num_rdofs = self.bglobal_dofs_ng[i].shape[0]

            with fi.vector.localForm() as loc:
                loc.array[self.bglobal_dofs_ng[i] - self.boffsets_ng[i]] = \
                    rx.array_r[rdof_offset:(rdof_offset + num_rdofs)]

            fi.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            rdof_offset += num_rdofs
