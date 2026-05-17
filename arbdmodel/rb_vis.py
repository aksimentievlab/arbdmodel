"""rb_vis.py — VMD visualization helper for ARBD rigid-body trajectories.

Usage — with any model::

    from arbdmodel import ArbdModel, RBContactModel
    from arbdmodel.rb_vis import ArbdVis

    # Works with RBContactModel (DiffusiveRigidBodyType — structure files known)
    model = RBContactModel(...)
    vis = ArbdVis(model, traj_path="output/sim.rb-traj", output_dir="vis/")
    vis.write()

    # Also works with a plain ArbdModel that has DiffusiveRigidBodyType objects
    model = ArbdModel(...)
    vis = ArbdVis(model, traj_path="output/sim.rb-traj", output_dir="vis/")
    vis.write()

    # Plain RigidBodyType (no structure file) — supply structure paths explicitly
    model = ArbdModel(...)   # uses bare RigidBodyType, not DiffusiveRigidBodyType
    vis = ArbdVis(
        model,
        traj_path="output/sim.rb-traj",
        output_dir="vis/",
        rb_types=[("myRBtype", "/path/to/struct")],  # overrides model lookup
    )
    vis.write()

Low-level usage (no model object at all)::

    vis = ArbdVis(
        traj_path="output/sim.rb-traj",
        output_dir="vis/",
        rb_types=[
            ("nanoparticle", "/path/to/nanoparticle"),  # stem, no extension
            ("capsid",       "/path/to/capsid.psf"),    # or full path
        ],
        static_pdbs=[("/path/to/pore.psf", "/path/to/pore.pdb")],
    )
    vis.write()
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Union

from .logger import get_resource_path, logger

# ── Display presets ────────────────────────────────────────────────────────────
# Each preset is a block of TCL that configures VMD aesthetics.
# 'publication' targets a white-background, shadow+AO render suitable for figures.
# 'minimal' is a plain dark background with no extra rendering cost.

_DISPLAY_PRESETS = {
    "publication": """\
color Display Background white
display projection   Perspective
display depthcue     on
display cuedensity   0.22
display backgroundgradient off
display shadows      on
display ambientocclusion on
display aoambient    1.6
display aodirect     0.0
display dof_focaldist 2.0
display dof_fnumber  1200
display rendermode   GLSL
display resize 2160 2160
axes location Off
mol material AOChalky
""",
    "minimal": """\
color Display Background black
display projection Orthographic
display depthcue   off
display shadows    off
display ambientocclusion off
axes location Off
""",
}


def _struct_stem_from_rb_type(rb_type) -> Optional[str]:
    """Return the structure file stem for a RigidBodyType, or None if unavailable.

    ``DiffusiveRigidBodyType`` objects carry ``aligned_psf`` / ``aligned_pdb``
    set during construction.  Plain ``RigidBodyType`` objects do not.  The TCL
    ``loadTrajectory`` proc tries ``<stem>.psf``/``<stem>.pdb`` then
    ``<stem>.xyz``, so we strip the suffix here.
    """
    for attr in ("aligned_psf", "aligned_pdb"):
        path = getattr(rb_type, attr, None)
        if path is not None:
            return str(Path(path).with_suffix(""))
    return None


class ArbdVis:
    """Generate VMD visualization scripts for an ARBD rigid-body simulation.

    Parameters
    ----------
    traj_path : str or Path
        Path (or glob) to the ``.rb-traj`` file(s) produced by ARBD.
    output_dir : str or Path
        Directory where ``launcher.tcl`` and ``run_vis.sh`` will be written.
        Created automatically if it does not exist.
    model : ArbdModel or RBContactModel, optional
        Any ``ArbdModel`` (or subclass such as ``RBContactModel``).  RB type
        information is read from the model automatically:
        ``RBContactModel._diffusible_rb_types`` (preferred) or
        ``ArbdModel.rigid_body_index`` (fallback for plain models).
        Types that carry structure files (``DiffusiveRigidBodyType``) are
        picked up automatically; plain ``RigidBodyType`` objects have no
        structure file and must be supplemented via *rb_types*.
        Either *model* or *rb_types* must be supplied.
    rb_types : list of (name, struct_path), optional
        Explicit list of ``(key_root, structure_path)`` pairs.  *struct_path*
        may be a stem (no extension) — the TCL loader will try
        ``<stem>.psf``/``<stem>.pdb`` then ``<stem>.xyz`` — or a full path
        with extension.
    static_pdbs : list of (psf_path, pdb_path), optional
        Static background structures to load as fixed molecules.
    dcd_path : str or Path, optional
        Particle-level DCD file to load onto the dummy mol alongside the RB
        trajectory (enables simultaneous viewing of particle and RB motion).
    skip : int
        Trajectory stride passed to VMD (default 1).
    beg : int
        First frame to load (default 0).
    end : int
        Last frame to load; -1 means all (default -1).
    vmd_path : str or Path
        Path to the VMD executable (default ``/Common/linux/bin/vmd``).
    display : str or None
        Display preset name — ``'publication'`` (default) or ``'minimal'``.
        Pass ``None`` to omit display configuration entirely.
    """

    def __init__(
        self,
        traj_path: Union[str, Path],
        output_dir: Union[str, Path],
        model=None,
        rb_types: Optional[list] = None,
        static_pdbs: Optional[list] = None,
        dcd_path: Optional[Union[str, Path]] = None,
        skip: int = 1,
        beg: int = 0,
        end: int = -1,
        vmd_path: Union[str, Path] = Path("/Common/linux/bin/vmd"),
        display: Optional[str] = "publication",
    ):
        if model is None and rb_types is None:
            raise ValueError("Provide either 'model' or 'rb_types'.")

        self.traj_path  = Path(traj_path)
        self.output_dir = Path(output_dir)
        self.dcd_path   = Path(dcd_path) if dcd_path else None
        self.skip       = skip
        self.beg        = beg
        self.end        = end
        self.vmd_path   = Path(vmd_path)
        self.display    = display

        # ── Resolve RB types ──────────────────────────────────────────────
        if model is not None:
            self._rb_types = self._rb_types_from_model(model)
        else:
            self._rb_types = list(rb_types)  # [(name, struct_path), ...]

        # ── Resolve static objects ────────────────────────────────────────
        if static_pdbs is not None:
            self._static_pdbs = [(Path(psf), Path(pdb)) for psf, pdb in static_pdbs]
        elif model is not None and hasattr(model, "static_objects"):
            self._static_pdbs = self._static_pdbs_from_model(model)
        else:
            self._static_pdbs = []

        # ── Package resource paths ────────────────────────────────────────
        self._procs_tcl = get_resource_path("arbd_vis.procs.tcl")
        self._dummy_psf = get_resource_path("dummy.psf")

    # ── Public interface ───────────────────────────────────────────────────────

    def write(self) -> Path:
        """Write ``launcher.tcl`` and ``run_vis.sh`` to *output_dir*.

        Returns the path to ``run_vis.sh``.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        launcher = self._write_launcher_tcl()
        bash     = self._write_bash(launcher)
        logger.info(f"ArbdVis: visualization scripts written to {self.output_dir}")
        logger.info(f"  launcher : {launcher}")
        logger.info(f"  runner   : {bash}")
        return bash

    def run(self) -> None:
        """Write scripts and immediately launch VMD."""
        bash = self.write()
        logger.info(f"ArbdVis: launching VMD via {bash}")
        subprocess.run(["bash", str(bash)], check=True)

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _rb_types_from_model(model) -> list:
        """Extract (name, struct_stem) pairs from any ArbdModel or RBContactModel.

        Resolution order for the list of RigidBodyType objects:

        1. ``model._diffusible_rb_types`` — present on ``RBContactModel``, gives
           ``DiffusiveRigidBodyType`` objects that always carry ``aligned_psf``.
        2. ``model.rigid_body_index`` — present on any ``ArbdModel`` after
           ``_count_types()``, maps ``{name: type_object}``.  Works for both
           ``DiffusiveRigidBodyType`` (has ``aligned_psf``) and plain
           ``RigidBodyType`` (no structure file; user must supply ``rb_types``
           manually for those).

        Types with no structure file are skipped with a warning.
        """
        result = []

        # ── Path 1: RBContactModel ────────────────────────────────────────
        diffusible = getattr(model, "_diffusible_rb_types", None)
        if diffusible:
            for rb_type in diffusible:
                struct = _struct_stem_from_rb_type(rb_type)
                if struct is not None:
                    result.append((rb_type.name, struct))
            return result

        # ── Path 2: plain ArbdModel — read rigid_body_index ──────────────
        rb_index = getattr(model, "rigid_body_index", None)
        if rb_index is None:
            # _count_types() may not have been called yet; trigger it
            try:
                model._count_types()
                rb_index = getattr(model, "rigid_body_index", {})
            except Exception as exc:
                logger.warning(f"ArbdVis: could not read RB types from model ({exc}); "
                               "supply rb_types manually")
                return result

        for name, rb_type in (rb_index or {}).items():
            struct = _struct_stem_from_rb_type(rb_type)
            if struct is not None:
                result.append((name, struct))
            else:
                logger.warning(
                    f"ArbdVis: RB type '{name}' is a plain RigidBodyType with no "
                    "structure file — add it via the rb_types argument: "
                    f"rb_types=[('{name}', '/path/to/struct')]"
                )

        if not result:
            logger.warning(
                "ArbdVis: no RB types with structure files found in model; "
                "supply rb_types manually"
            )
        return result

    @staticmethod
    def _static_pdbs_from_model(model) -> list:
        """Extract (psf, pdb) path pairs from StaticObject list on the model."""
        result = []
        for obj in getattr(model, "static_objects", []):
            sp = getattr(obj, "structure_path", None)
            if sp is None:
                continue
            sp = Path(sp)
            # structure_path may be a PSF; look for the co-located PDB
            psf = sp if sp.suffix == ".psf" else sp.with_suffix(".psf")
            pdb = sp.with_suffix(".pdb")
            if psf.exists() and pdb.exists():
                result.append((psf, pdb))
            else:
                logger.warning(
                    f"ArbdVis: static object '{obj.name}' — "
                    f"could not find PSF+PDB pair at {psf} / {pdb}"
                )
        return result

    def _write_launcher_tcl(self) -> Path:
        out_path = self.output_dir / "launcher.tcl"

        lines = [
            "## launcher.tcl — auto-generated by arbdmodel ArbdVis",
            "## Edit run_vis.sh to change skip/beg/end; do not edit this file directly.",
            "",
            f"source {{{self._procs_tcl}}}",
            "",
            f"set skip {self.skip}",
            f"set beg  {self.beg}",
            f"set end  {self.end}",
            "",
            "## ── Dummy mol: provides the frame counter VMD needs ─────────────────",
            f"set attachID [mol new {{{self._dummy_psf}}}]",
            "mol off $attachID",
            "",
        ]

        # ── Optional particle DCD ─────────────────────────────────────────
        if self.dcd_path is not None:
            lines += [
                "## ── Particle DCD (all-atom / CG trajectory) ─────────────────────",
                f"mol addfile {{{self.dcd_path}}} waitfor all",
                "",
            ]

        # ── One loadTrajectory call per RB type ───────────────────────────
        lines += [
            "## ── Rigid body trajectories (one call per type) ─────────────────────",
            "set rbIDs {}",
        ]
        for name, struct in self._rb_types:
            lines.append(
                f"set rbIDs [concat $rbIDs [loadTrajectory "
                f"{{{struct}}} "
                f"{{{self.traj_path}}} "
                f"$attachID $skip $beg $end "
                f"{{{name}}}]]"
            )
        lines.append("")

        # ── Default RB representation ─────────────────────────────────────
        lines += [
            "## ── Default representation for each RB instance ─────────────────────",
            "foreach rbID $rbIDs {",
            "    mol modstyle    0 $rbID QuickSurf 1.0 0.5 1.9 1.0",
            "    mol modmaterial 0 $rbID AOChalky",
            "    mol modcolor    0 $rbID Molecule",
            "}",
            "",
        ]

        # ── Static background objects ─────────────────────────────────────
        if self._static_pdbs:
            lines.append("## ── Static background objects ───────────────────────────────────────")
            for i, (psf, pdb) in enumerate(self._static_pdbs):
                lines += [
                    f"mol load psf {{{psf}}} pdb {{{pdb}}}",
                    f"mol modstyle    0 top QuickSurf 1.0 0.5 1.9 1.0",
                    f"mol modmaterial 0 top AOChalky",
                    f"mol modcolor    0 top ColorID {i}",
                ]
            lines.append("")

        # ── Display preset ────────────────────────────────────────────────
        if self.display and self.display in _DISPLAY_PRESETS:
            lines += [
                f"## ── Display preset: {self.display} ─────────────────────────────────",
                _DISPLAY_PRESETS[self.display],
            ]
        elif self.display and self.display not in _DISPLAY_PRESETS:
            logger.warning(
                f"ArbdVis: unknown display preset '{self.display}'; "
                f"valid options are {list(_DISPLAY_PRESETS)}"
            )

        lines += [
            "## Return focus to dummy mol so the frame slider controls everything",
            "mol top $attachID",
            "display resetview",
        ]

        with open(out_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return out_path

    def _write_bash(self, launcher_tcl: Path) -> Path:
        out_path = self.output_dir / "run_vis.sh"
        content = f"""\
#!/bin/bash
# run_vis.sh — auto-generated by arbdmodel ArbdVis
# Launch VMD with the ARBD rigid-body trajectory.
#
# To adjust stride or frame range, edit skip/beg/end in launcher.tcl,
# or re-run ArbdVis.write() with different parameters.

set -e
{self.vmd_path} -e {launcher_tcl}
"""
        with open(out_path, "w") as fh:
            fh.write(content)
        os.chmod(out_path, 0o755)
        return out_path
