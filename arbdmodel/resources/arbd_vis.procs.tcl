## arbd_vis.procs.tcl
## Static TCL procedure library for visualizing ARBD rigid-body trajectories in VMD.
## Generated once as a package resource — do not edit the copy inside the installed package.
##
## Public API:
##   loadTrajectory rbvmd files ?attachID? ?skip? ?beg? ?end? ?keyFilter?
##   loadTrajectoryRbFrame files ?attachID? ?skip? ?beg? ?end?
##   smoothRot frames
##   centerAll ref
##   frameChange varname ID rw          (set as a trace callback; do not call directly)
##   frameChangeRbFrame varname ID rw   (set as a trace callback; do not call directly)
##
## Utility:
##   sortFileGlob fileGlob
##   numframes dcdFile ...
##   parseRigidBodyTrajectoryFiles files ?skip? ?beg? ?end?
##   parseRigidBodyTrajectory file ?skip? ?counter? ?beg? ?end?
##   calcTransInv
##   matrixToQuaternion m
##   quaternionToMatrix quat
##   quaternionAndTransToMatrix q r
##   smooth4by4RotationMatrices frames rots

## ---------------------------------------------------------------------------
## sortFileGlob -- sort files matched by a glob by length then lexicographically
## ---------------------------------------------------------------------------
proc sortFileGlob {fileGlob} {
    set files [glob $fileGlob]
    set sortCmd {{f1 f2} {
        set l1 [string length $f1]
        set l2 [string length $f2]
        set r [expr {($l1 > $l2) - ($l1 < $l2)}]
        if {$r == 0} { set r [string compare $f1 $f2] }
        return $r
    }}
    lsort -command "apply {$sortCmd}" $files
}

## ---------------------------------------------------------------------------
## numframes -- count total frames in one or more DCD files using catdcd
## ---------------------------------------------------------------------------
proc numframes {args} {
    set r 0
    foreach dcd $args {
        set n [exec catdcd -num $dcd | awk "/^Total frames:/ \{ print \$3 \}"]
        set r [expr {$r + $n}]
    }
    return $r
}

## ---------------------------------------------------------------------------
## dcd -- load DCD file(s) onto molid, respecting beg/skip/end
## ---------------------------------------------------------------------------
proc dcd {dcdGlob args} {
    set molid ""
    set defaults { {skip 1} {beg 0} {end 0} }
    foreach {var val} [join $defaults] { set $var $val }
    foreach {var val} [join $args]     { set $var $val }

    if { [string equal $molid top] || [string equal $molid ""] } {
        set molid [molinfo top]
    }

    set startFrames [molinfo $molid get numframes]
    set framesLeft 1

    foreach glob $dcdGlob {
        set dcds [sortFileGlob $glob]
        foreach dcd $dcds {
            if { $framesLeft == 0 } { break }
            if { ![file exists $dcd] } { continue }
            set dcdFrames [numframes $dcd]
            puts "reading $dcd ($dcdFrames frames)"

            if { $beg > 0 } { set animateBeg $beg } else { set animateBeg 0 }
            set animateEnd ""
            if { $dcdFrames > $end && $end > 0 } {
                set animateEnd $end
                set framesLeft 0
            }

            set beg [expr {$beg - $dcdFrames}]
            set end [expr {$end - $dcdFrames}]
            if { $animateBeg >= $dcdFrames } { continue }
            set beg [expr {$skip - 1 - int(fmod(($dcdFrames - $animateBeg), $skip))}]

            set animateBeg "beg $animateBeg"
            if { $animateEnd > 0 } { set animateEnd "end $animateEnd" }
            uplevel "animate read dcd $dcd skip $skip $animateBeg $animateEnd waitfor all $molid"
        }
    }
    expr { [molinfo $molid get numframes] - $startFrames }
}

## ---------------------------------------------------------------------------
## loadTrajectory -- load an ARBD rigid-body trajectory into VMD
##
##   rbvmd     : structure file for the rigid body (path without extension, or
##               full path with extension).  Tries <rbvmd>.psf+<rbvmd>.pdb,
##               then <rbvmd>.xyz, then <rbvmd> verbatim.
##   files     : glob or list of .rb-traj files
##   attachID  : VMD mol ID to attach the frame-change trace to (default: top)
##   skip      : stride (default 1)
##   beg       : first frame (default 0)
##   end       : last frame, -1 = all (default -1)
##   keyFilter : if non-empty, only keys whose root matches this string are
##               loaded.  Use one loadTrajectory call per RB type.
##               (default "" = load all keys)
##
## Returns a list of VMD mol IDs, one per RB instance loaded.
## ---------------------------------------------------------------------------
proc loadTrajectory {rbvmd files {attachID top} {skip 1} {beg 0} {end -1} {keyFilter ""}} {
    variable trans
    variable trans_orig
    variable trans_inv
    variable molToKey
    variable lastFrame
    variable rigidBodyIDs

    if { [string equal $attachID top] } { set attachID [molinfo top] }
    
    array set trans [parseRigidBodyTrajectoryFiles $files $skip $beg $end $keyFilter]
    set keys [array names trans]

    if { [llength $keys] == 0 } {
        puts "WARNING: loadTrajectory: no keys found in $files (keyFilter='$keyFilter')"
        return {}
    }

    set rbFrames  [llength $trans([lindex $keys 0])]
    set numframes [molinfo $attachID get numframes]

    if { $rbFrames < $numframes } {
        puts "WARNING: Read $rbFrames rigid body frames < $numframes all-atom frames; padding last frame"
    }
    while { $rbFrames < $numframes } {
        foreach key [array names trans] { lappend trans($key) [lindex $trans($key) end] }
        incr rbFrames
    }
    if { $rbFrames < $numframes } {
        error "loadTrajectory: $rbFrames RB frames < $numframes all-atom frames after padding"
    }

    array set trans_orig [array get trans]
    calcTransInv

    set topID [molinfo top]
    set newIDs {}

    if { [catch {
        ## Initialise lastFrame safely
        if { [info exists ::vmd_frame($attachID)] } {
            set lastFrame $::vmd_frame($attachID)
        } else {
            set lastFrame 0
            set ::vmd_frame($attachID) 0
        }

        foreach key [lsort -dictionary [array names trans]] {
            if [info exists molToKey($key)] continue

            ## Load structure: try PSF/PDB pair, then XYZ, then verbatim path
            if { [file exists ${rbvmd}.psf] && [file exists ${rbvmd}.pdb] } {
                set ID [mol new ${rbvmd}.psf]
                mol addfile ${rbvmd}.pdb
            } elseif { [file exists ${rbvmd}.xyz] } {
                set ID [mol new ${rbvmd}.xyz waitfor all]
            } elseif { [file exists $rbvmd] } {
                set ID [mol new $rbvmd waitfor all]
            } else {
                error "Cannot find structure for '$rbvmd' (.psf/.pdb, .xyz, or verbatim)"
            }

            molinfo $ID set {a b c} [molinfo $attachID get {a b c}]
            lappend newIDs $ID
            lappend rigidBodyIDs $ID
            set molToKey($ID) $key
            set molToKey($key) $ID
            [atomselect $ID all] move [lindex $trans($key) $lastFrame]
        }

        ## Remove any existing trace, then set ours
        variable ::vmd_frame
        foreach elem [trace info variable ::vmd_frame($attachID)] {
            foreach {opList cmd} $elem {
                trace remove variable ::vmd_frame($attachID) $opList $cmd
            }
        }
        trace variable ::vmd_frame($attachID) w frameChange

    } errMsg]} {
        puts "WARNING: loadTrajectory failed: $errMsg"
    }

    mol top $topID
    return $newIDs
}

## ---------------------------------------------------------------------------
## loadTrajectoryRbFrame -- load trajectory in the reference frame of one RB
## ---------------------------------------------------------------------------
proc loadTrajectoryRbFrame {files {attachID top} {skip 1} {beg 0} {end -1}} {
    variable trans
    variable trans_orig
    variable trans_inv
    variable molToKey
    variable lastFrame
    variable rigidBodyIDs
    variable centerRbID
    set rigidBodyIDs ""

    if { [string equal $attachID top] } { set attachID [molinfo top] }

    array set trans [parseRigidBodyTrajectoryFiles $files $skip $beg $end]
    set keys        [array names trans]
    set rbFrames    [llength $trans([lindex $keys 0])]
    set numframes   [molinfo $attachID get numframes]
    if { $rbFrames < $numframes } {
        error "loadTrajectoryRbFrame: Read $rbFrames RB frames < $numframes all-atom frames"
    }

    array set trans_orig [array get trans]
    calcTransInv

    set topID [molinfo top]
    if { [catch {
        set lastFrame $::vmd_frame($attachID)
        foreach key [array names trans] {
            set ID [mol new dummy.psf]
            molinfo $ID set {a b c} [molinfo $attachID get {a b c}]
            lappend rigidBodyIDs $ID
            set molToKey($ID) $key
        }

        if { ![info exists centerRbID] } { set centerRbID [lindex $rigidBodyIDs 0] }

        if { [catch {
            set rbID $centerRbID
            set key  $molToKey($rbID)
            foreach tID [molinfo list] {
                if { [molinfo $tID get active] && [molinfo $tID get numframes] > 1 } {
                    set sel [atomselect $tID all]
                    frameLoop frame molid $tID {
                        $sel frame $frame
                        $sel move [lindex $trans_inv($key) $frame]
                    }
                }
            }
        }]} { puts "WARNING: failed to initialize RB frame" }

        variable ::vmd_frame
        foreach elem [trace info variable ::vmd_frame($attachID)] {
            foreach {opList cmd} $elem {
                trace remove variable ::vmd_frame($attachID) $opList $cmd
            }
        }
        trace variable ::vmd_frame($attachID) w frameChangeRbFrame
    }]} {
        puts "WARNING: loadTrajectoryRbFrame: failed to set trace"
    }
    mol top $topID
}

## ---------------------------------------------------------------------------
## Callbacks -- called automatically by VMD when the frame slider moves
## ---------------------------------------------------------------------------
proc ::frameChangeCallback {frame} {}

proc frameChange {varname ID rw} {
    variable rigidBodyIDs
    variable trans
    variable trans_inv
    variable molToKey
    variable lastFrame
    set frame $::vmd_frame($ID)
    if { [catch {
        foreach rbID $rigidBodyIDs {
            set key $molToKey($rbID)
            set sel [atomselect $rbID all]
            $sel move [lindex $trans_inv($key) $lastFrame]
            $sel move [lindex $trans($key)     $frame]
        }
    }]} { puts "WARNING: frameChange: failed to update rigid body positions" }
    catch { ::frameChangeCallback $frame }
    set lastFrame $frame
}

proc frameChangeRbFrame {varname ID rw} {
    variable rigidBodyIDs
    variable trans
    variable trans_inv
    variable molToKey
    variable lastFrame
    variable centerRbID
    set frame $::vmd_frame($ID)

    if { [catch {
        set key $molToKey($centerRbID)
        foreach tID $rigidBodyIDs {
            [atomselect $tID all] move [lindex $trans($key) $lastFrame]
        }
    }]} { puts "WARNING: frameChangeRbFrame: undo step failed" }

    if { [catch {
        foreach rbID $rigidBodyIDs {
            set key $molToKey($rbID)
            [atomselect $rbID all] move \
                [transmult [lindex $trans($key) $frame] [lindex $trans_inv($key) $lastFrame]]
        }
    }]} { puts "WARNING: frameChangeRbFrame: per-RB transform failed" }

    if { [catch {
        set key $molToKey($centerRbID)
        foreach tID $rigidBodyIDs {
            [atomselect $tID all] move [lindex $trans_inv($key) $frame]
        }
    }]} { puts "WARNING: frameChangeRbFrame: global transform failed" }

    set lastFrame $frame
}

## ---------------------------------------------------------------------------
## parseRigidBodyTrajectoryFiles -- parse multiple .rb-traj files
##   keyFilter: if non-empty, only keep keys whose root matches this string
## ---------------------------------------------------------------------------
proc parseRigidBodyTrajectoryFiles {files {skip 1} {beg 0} {end -1} {keyFilter ""}} {
    set file  [lindex $files 0]
    set files [lrange $files 1 end]

    set counter 0
    lassign [parseRigidBodyTrajectory $file $skip $counter $beg $end $keyFilter] counter tmp
    array set trans $tmp
    set keys [array names trans]

    foreach file $files {
        lassign [parseRigidBodyTrajectory $file $skip $counter $beg $end $keyFilter] counter tmp
        array set newTrans $tmp
        if { [lsort $keys] != [lsort [array names newTrans]] } {
            puts stderr "parseRigidBodyTrajectoryFiles: $file does not share the same rigid bodies"
            puts stderr "  expected: [lsort $keys]"
            puts stderr "  got:      [lsort [array names newTrans]]"
            exit 1
        }
        foreach key $keys {
            set trans($key) [join "{$trans($key)} {$newTrans($key)}"]
        }
    }
    return [array get trans]
}

## ---------------------------------------------------------------------------
## parseRigidBodyTrajectory -- parse a single .rb-traj file into a trans array
##   keyFilter: if non-empty, skip keys whose root does not match
## ---------------------------------------------------------------------------
proc parseRigidBodyTrajectory {file {skip 1} {counter 0} {beg 0} {end -1} {keyFilter ""}} {
    array set trans      ""
    array set keyCounter ""

    set ch [open $file]
    while { [gets $ch line] > 0 } {
        ## skip comment/header lines
        if { [regexp {^\W*#} $line] } { continue }

        lassign [lindex $line 1] key

        ## apply keyFilter: key format is  "keyRoot#index"
        if { $keyFilter ne "" } {
            regexp {([^#]*)#?.*} $key --> keyRoot
            if { $keyRoot ne $keyFilter } { continue }
        }

        if { ![info exists trans($key)] }      { set trans($key)      "" }
        if { ![info exists keyCounter($key)] } { set keyCounter($key) $counter } \
        else                                   { incr keyCounter($key) }

        if { fmod($keyCounter($key), $skip) > 0.1 || $keyCounter($key) < $beg } { continue }
        if { $end >= 0 && $keyCounter($key) > $end + 2*$skip } { break }
        if { $end >= 0 && $keyCounter($key) > $end }           { continue }

        set m ""
        lappend m [join "[lrange $line 5 7] 0"]
        lappend m [join "[lrange $line 8 10] 0"]
        lappend m [join "[lrange $line 11 13] 0"]
        lappend m {0 0 0 1}
        set m [transmult [transoffset [lrange $line 2 4]] $m]
        lappend trans($key) "$m"
    }
    close $ch
    return "[lindex [array get keyCounter] 1] {[array get trans]}"
}

## ---------------------------------------------------------------------------
## calcTransInv -- precompute inverse transformation matrices
## ---------------------------------------------------------------------------
proc calcTransInv {} {
    variable trans
    variable trans_inv
    array set trans_inv {}
    foreach key [array names trans] {
        set trans_inv($key) ""
        foreach m $trans($key) {
            if { [catch { lappend trans_inv($key) [measure inverse $m] }] } {
                error "calcTransInv: could not invert matrix '$m'"
            }
        }
    }
}

## ---------------------------------------------------------------------------
## smoothRot -- smooth rigid body rotation using quaternion averaging
## ---------------------------------------------------------------------------
proc smoothRot {frames} {
    variable rigidBodyIDs
    variable trans_orig
    variable trans
    variable trans_inv
    variable molToKey
    variable lastFrame
    variable centerRbID

    if { [catch {
        foreach rbID $rigidBodyIDs {
            set key $molToKey($rbID)
            if { ![info exists centerRbID] || $rbID != $centerRbID } {
                [atomselect $rbID all] move [lindex $trans_inv($key) $lastFrame]
            }
        }
    }]} { puts "WARNING: smoothRot: failed to reset positions" }

    if { [catch {
        foreach rbID $rigidBodyIDs {
            set key $molToKey($rbID)
            set trans($key) [smooth4by4RotationMatrices $frames $trans_orig($key)]
        }
    }]} { puts "WARNING: smoothRot: failed to smooth" }

    calcTransInv

    if { [catch {
        foreach rbID $rigidBodyIDs {
            set key $molToKey($rbID)
            if { ![info exists centerRbID] || $rbID != $centerRbID } {
                [atomselect $rbID all] move [lindex $trans($key) $lastFrame]
            }
        }
    }]} { puts "WARNING: smoothRot: failed to restore positions" }
}

## ---------------------------------------------------------------------------
## centerAll -- translate all mols so that ref stays centered each frame
## ---------------------------------------------------------------------------
proc centerAll {ref} {
    variable rigidBodyIDs
    variable trans
    variable trans_inv
    variable molToKey
    variable lastFrame

    set ID [$ref molid]

    if { [catch {
        foreach rbID $rigidBodyIDs {
            set key $molToKey($rbID)
            [atomselect $rbID all] move [lindex $trans_inv($key) $lastFrame]
        }
    }]} { puts "WARNING: centerAll: failed to reset positions" }

    array set newTrans ""
    frameLoop f molid $ID {
        $ref frame $f
        set v [vecinvert [measure center $ref]]
        foreach rbID $rigidBodyIDs {
            set key $molToKey($rbID)
            lappend newTrans($key) [transmult [transoffset $v] [lindex $trans($key) $f]]
        }
    }
    foreach rbID $rigidBodyIDs {
        set key $molToKey($rbID)
        set trans($key) $newTrans($key)
    }

    calcTransInv

    if { [catch {
        foreach rbID $rigidBodyIDs {
            set key $molToKey($rbID)
            [atomselect $rbID all] move [lindex $trans($key) $lastFrame]
        }
    }]} { puts "WARNING: centerAll: failed to restore positions" }

    set all [atomselect $ID all]
    frameLoop f molid $ID {
        $ref frame $f
        $all frame $f
        $all moveby [vecinvert [measure center $ref]]
    }
}

## ---------------------------------------------------------------------------
## Quaternion / matrix utilities
## ---------------------------------------------------------------------------
proc matrixToQuaternion {m} {
    lassign [lindex $m 0] Axx Axy Axz x1
    lassign [lindex $m 1] Ayx Ayy Ayz x2
    lassign [lindex $m 2] Azx Azy Azz x3

    set d1 [expr {1 + $Axx + $Ayy + $Azz}]
    set d1 [expr {($d1 < 0) ? 0 : 0.5*sqrt($d1)}]
    set d2 [expr {1 + $Axx - $Ayy - $Azz}]
    set d2 [expr {($d2 < 0) ? 0 : 0.5*sqrt($d2)}]

    if { $d1 > $d2 } {
        set q4 $d1
        set q1 [expr {($Azy - $Ayz)*0.25/$q4}]
        set q2 [expr {($Axz - $Azx)*0.25/$q4}]
        set q3 [expr {($Ayx - $Axy)*0.25/$q4}]
    } else {
        set q1 $d2
        set q2 [expr {($Axy + $Ayx)*0.25/$q1}]
        set q3 [expr {($Axz + $Azx)*0.25/$q1}]
        set q4 [expr {($Azy - $Ayz)*0.25/$q1}]
    }
    return "$q1 $q2 $q3 $q4"
}

proc quaternionToMatrix {quat} {
    lassign $quat q1 q2 q3 q4
    return [list \
        [list [expr {1-2*($q2*$q2+$q3*$q3)}] [expr {2*($q1*$q2-$q3*$q4)}] [expr {2*($q1*$q3+$q2*$q4)}]] \
        [list [expr {2*($q1*$q2+$q3*$q4)}]   [expr {1-2*($q1*$q1+$q3*$q3)}] [expr {2*($q2*$q3-$q1*$q4)}]] \
        [list [expr {2*($q1*$q3-$q2*$q4)}]   [expr {2*($q1*$q4+$q2*$q3)}]   [expr {1-2*($q2*$q2+$q1*$q1)}]] \
    ]
}

proc quaternionAndTransToMatrix {q r} {
    lassign $r r1 r2 r3
    lassign $q q1 q2 q3 q4
    return "{[expr {1-2*($q2*$q2+$q3*$q3)}] [expr {2*($q1*$q2-$q3*$q4)}] [expr {2*($q1*$q3+$q2*$q4)}] $r1}
            {[expr {2*($q1*$q2+$q3*$q4)}]   [expr {1-2*($q1*$q1+$q3*$q3)}] [expr {2*($q2*$q3-$q1*$q4)}] $r2}
            {[expr {2*($q1*$q3-$q2*$q4)}]   [expr {2*($q1*$q4+$q2*$q3)}]   [expr {1-2*($q2*$q2+$q1*$q1)}] $r3}
            {0 0 0 1}"
}

proc smooth4by4RotationMatrices {frames rots} {
    set newRots ""
    set qs ""
    set rs ""
    foreach m $rots {
        lappend rs "[lindex $m 0 3] [lindex $m 1 3] [lindex $m 2 3]"
        lappend qs [matrixToQuaternion $m]
    }

    set numFrames [llength $rots]
    set rStack    [lrange $rs 0 [expr {$frames - 1}]]
    set qStack    [lrange $qs 0 [expr {$frames - 1}]]
    set newR      [eval "vecadd $rStack"]
    set newQ      [eval "vecadd $qStack"]

    for {set f 0} {$f < $numFrames} {incr f} {
        if { $f > $frames } {
            set newR   [vecsub $newR [lindex $rStack 0]]
            set rStack [lrange $rStack 1 end]
            set newQ   [vecsub $newQ [lindex $qStack 0]]
            set qStack [lrange $qStack 1 end]
        }
        if { $f < $numFrames - $frames } {
            set r [lindex $rs [expr {$f + $frames}]]
            if { [catch { set newR [vecadd $newR $r] }] } { puts "smooth4by4: failed adding r '$r'" }
            lappend rStack $r
            set q [lindex $qs [expr {$f + $frames}]]
            if { [catch { set newQ [vecadd $newQ $q] }] } { puts "smooth4by4: failed adding q '$q'" }
            lappend qStack $q
        }
        if { [catch {
            lappend newRots [quaternionAndTransToMatrix \
                [vecnorm $newQ] \
                [vecscale [expr {1.0 / [llength $rStack]}] $newR]]
        }] } { puts "smooth4by4: failed at frame $f" }
    }
    return $newRots
}
