################################################################################
# Headless Multi-Region DFX Pblock Floorplanner
# Automatically packs multiple varying-size Pblocks without overlap.
################################################################################

proc generate_multi_dfx_pblocks {pblock_configs_list} {
    puts "================================================================"
    puts " Starting Automated Multi-Region DFX Floorplanning Pipeline"
    puts "================================================================"

    # --- 1. Query device fabric extents for all three resource coordinate systems ---

    # SLICE sites (contain LUTs): filter by X=0 column for max Y, Y=0 row for max X
    set slice_x0 [get_sites -filter {SITE_TYPE =~ "SLICE*" && NAME =~ "SLICE_X0Y*"}]
    if {[llength $slice_x0] == 0} {
        error "No SLICE sites found. Ensure a netlist/device is loaded in the current project."
    }
    set slice_y0 [get_sites -filter {SITE_TYPE =~ "SLICE*" && NAME =~ "SLICE_X*Y0"}]
    if {![regexp {SLICE_X0Y(\d+)}    [lindex [lsort -dictionary $slice_x0] end] -> max_device_y]} {
        error "Could not parse max SLICE Y from device."
    }
    if {![regexp {SLICE_X(\d+)Y0}    [lindex [lsort -dictionary $slice_y0] end] -> max_device_x]} {
        error "Could not parse max SLICE X from device."
    }

    # RAMB36 sites: iterate all to find max X and Y (coordinate system differs from SLICE)
    # Filter by NAME rather than SITE_TYPE — the site type string varies across device families,
    # but the site name prefix RAMB36_X#Y# is consistent.
    set max_ramb_x 0
    set max_ramb_y 0
    foreach site [get_sites -filter {NAME =~ "RAMB36_*"}] {
        if {[regexp {RAMB36_X(\d+)Y(\d+)} $site -> x y]} {
            if {$x > $max_ramb_x} { set max_ramb_x $x }
            if {$y > $max_ramb_y} { set max_ramb_y $y }
        }
    }

    # DSP48E2 sites: iterate all to find max X and Y (coordinate system differs from SLICE)
    set max_dsp_x 0
    set max_dsp_y 0
    foreach site [get_sites -filter {SITE_TYPE == "DSP48E2"}] {
        if {[regexp {DSP48E2_X(\d+)Y(\d+)} $site -> x y]} {
            if {$x > $max_dsp_x} { set max_dsp_x $x }
            if {$y > $max_dsp_y} { set max_dsp_y $y }
        }
    }

    puts "  Device SLICE  extent: X0..${max_device_x}  Y0..${max_device_y}"
    puts "  Device RAMB36 extent: X0..${max_ramb_x}  Y0..${max_ramb_y}"
    puts "  Device DSP48E2 extent: X0..${max_dsp_x}  Y0..${max_dsp_y}"

    # --- 2. Precompute SLICE/RAMB36/DSP48E2 X boundaries per clock region column ---
    # Clock region columns are natural DFX-legal X boundaries. Trying combinations from
    # narrowest (1 column) to widest (all columns) at each Y height finds the smallest
    # satisfying rectangle first, minimising pblock oversizing in the X dimension.
    # Collect unique CR X indices. get_clock_regions returns Tcl objects whose implicit
    # string form is device-version-dependent; get_property NAME gives the canonical
    # "CLOCKREGION_X#Y#" string reliably on all Vivado versions.
    # Collect unique CR X indices and cache one Y=0 object per column in a single pass.
    # Avoids a second get_clock_regions call (which warned because NAME may lack the
    # CLOCKREGION_ prefix, making an exact-string filter unreliable).
    set cr_x_indices {}
    foreach cr [get_clock_regions] {
        if {[regexp {X(\d+)Y(\d+)} [get_property NAME $cr] -> crx cry]} {
            # ONLY CONSIDER CR COLUMNS 1-4
            if {$crx >= 1 && $crx <= 4} {
                if {[lsearch -exact $cr_x_indices $crx] < 0} { lappend cr_x_indices $crx }
                # Prefer the Y=0 row as the column reference object
                if {$cry == 0 || ![info exists cr_ref($crx)]} { set cr_ref($crx) $cr }
            }
            # Cache a Y1 CR from within the allowed columns for the initial floor reference
            if {$crx >= 1 && $crx <= 4 && $cry == 1 && ![info exists cr_y1_ref]} { set cr_y1_ref $cr }
        }
    }
    set cr_x_indices [lsort -integer $cr_x_indices]
    set n_cr_cols [llength $cr_x_indices]
    puts "  Clock region columns: $cr_x_indices  ($n_cr_cols total)"

    # For each CR column, record the SLICE/RAMB36/DSP48E2 X range using the Y=0 reference
    # row (all rows in the same column share the same X span).
    for {set ci 0} {$ci < $n_cr_cols} {incr ci} {
        set crx    [lindex $cr_x_indices $ci]
        set ref_cr $cr_ref($crx)

        set col_sx_min($ci) 999999; set col_sx_max($ci) 0
        foreach s [get_sites -of_objects $ref_cr -filter {SITE_TYPE =~ "SLICE*"}] {
            if {[regexp {SLICE_X(\d+)} $s -> x]} {
                if {$x < $col_sx_min($ci)} { set col_sx_min($ci) $x }
                if {$x > $col_sx_max($ci)} { set col_sx_max($ci) $x }
            }
        }

        set r_sites [get_sites -of_objects $ref_cr -filter {NAME =~ "RAMB36_*"}]
        if {[llength $r_sites] > 0} {
            set col_rx_min($ci) 999999; set col_rx_max($ci) 0
            foreach s $r_sites {
                if {[regexp {RAMB36_X(\d+)} $s -> x]} {
                    if {$x < $col_rx_min($ci)} { set col_rx_min($ci) $x }
                    if {$x > $col_rx_max($ci)} { set col_rx_max($ci) $x }
                }
            }
        } else { set col_rx_min($ci) -1; set col_rx_max($ci) -1 }

        set d_sites [get_sites -of_objects $ref_cr -filter {SITE_TYPE == "DSP48E2"}]
        if {[llength $d_sites] > 0} {
            set col_dx_min($ci) 999999; set col_dx_max($ci) 0
            foreach s $d_sites {
                if {[regexp {DSP48E2_X(\d+)} $s -> x]} {
                    if {$x < $col_dx_min($ci)} { set col_dx_min($ci) $x }
                    if {$x > $col_dx_max($ci)} { set col_dx_max($ci) $x }
                }
            }
        } else { set col_dx_min($ci) -1; set col_dx_max($ci) -1 }
    }

    # --- 3. Track Y floors per resource type — start at bottom of CR row Y1 ---
    # Always compute the max site Y within CR row Y0 across all allowed columns.
    # This gives us the CR row height in each coordinate system, which is then
    # used as the search step so every trial boundary falls on a CR row edge.
    # A step that doesn't divide the CR row height causes SNAPPING_MODE to move
    # the SLICE ceiling to a different CR boundary than the proportional RAMB/DSP
    # ceilings, fragmenting the pblock into multiple rectangles.
    set max_y0_s 0; set max_y0_r 0; set max_y0_d 0
    for {set ci 0} {$ci < $n_cr_cols} {incr ci} {
        set ref_cr $cr_ref([lindex $cr_x_indices $ci])
        foreach s [get_sites -of_objects $ref_cr -filter {SITE_TYPE =~ "SLICE*"}] {
            if {[regexp {SLICE_X\d+Y(\d+)} $s -> y] && $y > $max_y0_s} { set max_y0_s $y }
        }
        foreach s [get_sites -of_objects $ref_cr -filter {NAME =~ "RAMB36_*"}] {
            if {[regexp {RAMB36_X\d+Y(\d+)} $s -> y] && $y > $max_y0_r} { set max_y0_r $y }
        }
        foreach s [get_sites -of_objects $ref_cr -filter {SITE_TYPE == "DSP48E2"}] {
            if {[regexp {DSP48E2_X\d+Y(\d+)} $s -> y] && $y > $max_y0_d} { set max_y0_d $y }
        }
    }
    # SEARCH HEIGHT STEP = CR ROW HEIGHT / 2 (e.g. 60/2=30 slice rows)
    set height_step [expr {($max_y0_s + 1) / 2}]
    puts "  CR row height: [expr {$max_y0_s + 1}] SLICE rows | step: ${height_step} SLICE rows"

    # Start floors at bottom of CR row Y1 if a suitable reference was found, else Y=0
    set slice_y_floor 0
    set ramb_y_floor  0
    set dsp_y_floor   0
    if {[info exists cr_y1_ref]} {
        set slice_y_floor [expr {$max_y0_s + 1}]
        set ramb_y_floor  [expr {$max_y0_r + 1}]
        set dsp_y_floor   [expr {$max_y0_d + 1}]
        puts "  Initial floors (CR row Y1): SLICE Y${slice_y_floor} | RAMB36 Y${ramb_y_floor} | DSP Y${dsp_y_floor}"
    } else {
        puts "  WARNING: No CR Y1 found in columns 1-4 — starting from Y=0"
    }

    foreach config $pblock_configs_list {
        set cell_name   [dict get $config cell]
        set pblock_name [dict get $config name]
        set req_luts    [dict get $config luts]
        set req_brams   [dict get $config brams]
        set req_dsps    [dict get $config dsps]

        puts "\n--> Processing: $pblock_name for cell $cell_name"
        puts "    Demands -> LUTs: $req_luts | BRAMs: $req_brams | DSPs: $req_dsps"

        # Clean up any pre-existing pblock with this name
        if {[get_pblocks -quiet $pblock_name] ne ""} { delete_pblocks $pblock_name }

        # --- 4. Search: expand Y height; at each height try all CR column X combinations ---
        # Combinations are ordered narrowest-first (1 column, then 2, ...) so the smallest
        # satisfying rectangle is accepted. Each trial recreates the pblock fresh because
        # resize_pblock -add accumulates and cannot be selectively cleared.
        set passed 0
        set slice_y_ceil [expr {$slice_y_floor + $height_step - 1}]

        while {$slice_y_ceil <= $max_device_y && !$passed} {
            # Scale the pblock height as a fraction of each resource type's *remaining* space,
            # so all three ranges always cover the same proportional physical region.
            # Using an absolute fraction (ceil/max) breaks for pblock 2+ because the snapped
            # floors drift from their proportional positions, causing Vivado to fragment the pblock.
            set slice_height [expr {$slice_y_ceil - $slice_y_floor}]
            set slice_remain [expr {$max_device_y  - $slice_y_floor}]
            set ramb_y_ceil  [expr {$ramb_y_floor  + int(double($slice_height) / $slice_remain * ($max_ramb_y - $ramb_y_floor))}]
            set dsp_y_ceil   [expr {$dsp_y_floor   + int(double($slice_height) / $slice_remain * ($max_dsp_y  - $dsp_y_floor))}]

            for {set width 1} {$width <= $n_cr_cols && !$passed} {incr width} {
                for {set ci_start 0} {$ci_start + $width - 1 < $n_cr_cols && !$passed} {incr ci_start} {
                    set ci_end [expr {$ci_start + $width - 1}]

                    # SLICE X span for this CR column range
                    set sx_start $col_sx_min($ci_start)
                    set sx_end   $col_sx_max($ci_end)

                    # RAMB36 X span: union across selected columns
                    set rx_start 999999; set rx_end 0; set has_ramb 0
                    for {set ci $ci_start} {$ci <= $ci_end} {incr ci} {
                        if {$col_rx_min($ci) >= 0} {
                            set has_ramb 1
                            if {$col_rx_min($ci) < $rx_start} { set rx_start $col_rx_min($ci) }
                            if {$col_rx_max($ci) > $rx_end}   { set rx_end   $col_rx_max($ci) }
                        }
                    }

                    # DSP48E2 X span: union across selected columns
                    set dx_start 999999; set dx_end 0; set has_dsp 0
                    for {set ci $ci_start} {$ci <= $ci_end} {incr ci} {
                        if {$col_dx_min($ci) >= 0} {
                            set has_dsp 1
                            if {$col_dx_min($ci) < $dx_start} { set dx_start $col_dx_min($ci) }
                            if {$col_dx_max($ci) > $dx_end}   { set dx_end   $col_dx_max($ci) }
                        }
                    }

                    # Recreate pblock fresh for this trial
                    startgroup
                    if {[get_pblocks -quiet $pblock_name] ne ""} { delete_pblocks $pblock_name }
                    endgroup
                    startgroup
                    create_pblock $pblock_name
                    set_property SNAPPING_MODE ON [get_pblocks $pblock_name]
                    if {[get_cells -quiet $cell_name] ne ""} {
                        add_cells_to_pblock [get_pblocks $pblock_name] [get_cells $cell_name]
                        set_property HD.RECONFIGURABLE 1 [get_cells $cell_name]
                    }

                    resize_pblock [get_pblocks $pblock_name] -add SLICE_X${sx_start}Y${slice_y_floor}:SLICE_X${sx_end}Y${slice_y_ceil}
                    if {$has_ramb} {
                        resize_pblock [get_pblocks $pblock_name] -add RAMB36_X${rx_start}Y${ramb_y_floor}:RAMB36_X${rx_end}Y${ramb_y_ceil}
                    }
                    if {$has_dsp} {
                        resize_pblock [get_pblocks $pblock_name] -add DSP48E2_X${dx_start}Y${dsp_y_floor}:DSP48E2_X${dx_end}Y${dsp_y_ceil}
                    }
                    endgroup


                    #after 3000
                    #update
                    #after 3000
                    #select_objects [get_pblocks]
                    #highlight_objects -color yellow [get_pblocks *]
                    #unhighlight_objects [get_pblocks *]

                    set avail_slices [llength [get_sites -of_objects [get_pblocks $pblock_name] -filter {SITE_TYPE =~ "SLICE*"}]]
                    set avail_luts   [expr {$avail_slices * 8}]
                    set avail_brams  [llength [get_sites -of_objects [get_pblocks $pblock_name] -filter {NAME =~ "RAMB36_*"}]]
                    set avail_dsps   [llength [get_sites -of_objects [get_pblocks $pblock_name] -filter {SITE_TYPE == "DSP48E2"}]]

                    puts "    CR_X${ci_start}..${ci_end} | SLICE X${sx_start}..${sx_end} Y${slice_y_floor}..${slice_y_ceil} | RAMB36 Y${ramb_y_floor}..${ramb_y_ceil} | DSP Y${dsp_y_floor}..${dsp_y_ceil}  ->  LUTs: $avail_luts/$req_luts  BRAMs: $avail_brams/$req_brams  DSPs: $avail_dsps/$req_dsps"

                    if {$avail_luts >= $req_luts && $avail_brams >= $req_brams && $avail_dsps >= $req_dsps} {
                        set passed 1
                    }
                }
            }

            if {!$passed} { incr slice_y_ceil $height_step }
        }

        if {$passed} {
            # --- 5. Finalise: apply DFX routing containment and read back snapped grid ---
            set_property CONTAIN_ROUTING 1  [get_pblocks $pblock_name]

            # Record the allocated pblock capacity in the global report dict
            global finn_pr_report
            if {![info exists finn_pr_report]} { set finn_pr_report [dict create] }
            dict set finn_pr_report $pblock_name pblock_capacity \
                [dict create luts $avail_luts brams $avail_brams dsps $avail_dsps]

            set snapped_grid [get_property GRID_RANGES [get_pblocks $pblock_name]]
            highlight_objects -color green [get_pblocks $pblock_name]
            puts "    SUCCESS: Allocated $pblock_name"
            puts "    Final Grid: $snapped_grid"

            # Parse the snapped max Y for each resource type separately so the next
            # pblock's floor is set precisely at the legal boundary for each.
            set max_snapped_slice_y $slice_y_floor
            set max_snapped_ramb_y  $ramb_y_floor
            set max_snapped_dsp_y   $dsp_y_floor
            foreach range $snapped_grid {
                if {[regexp {SLICE_X\d+Y\d+:SLICE_X\d+Y(\d+)}    $range -> y]} {
                    if {$y > $max_snapped_slice_y} { set max_snapped_slice_y $y }
                }
                if {[regexp {RAMB36_X\d+Y\d+:RAMB36_X\d+Y(\d+)}  $range -> y]} {
                    if {$y > $max_snapped_ramb_y}  { set max_snapped_ramb_y  $y }
                }
                if {[regexp {DSP48E2_X\d+Y\d+:DSP48E2_X\d+Y(\d+)} $range -> y]} {
                    if {$y > $max_snapped_dsp_y}   { set max_snapped_dsp_y   $y }
                }
            }

            set slice_y_floor [expr {$max_snapped_slice_y + 1}]
            set ramb_y_floor  [expr {$max_snapped_ramb_y  + 1}]
            set dsp_y_floor   [expr {$max_snapped_dsp_y   + 1}]

        } else {
            puts "    ERROR: Cannot fulfill resource demands for $pblock_name on remaining fabric!"
            return -code error "Floorplanning failed due to resource exhaustion."
        }
    }

    # --- 5. Export the resulting coordinates to an XDC file ---
    puts "\n================================================================"
    puts " Writing generated constraints to dfx_generated_floorplan.xdc"
    puts "================================================================"
    write_xdc -force -file dfx_generated_floorplan.xdc
}

################################################################################
# Resource Query Helper
# Reads per-cell LUT / BRAM / DSP utilisation from the open synthesis run.
################################################################################

proc query_cell_resources {cell_path} {
    # Returns a dict {luts N brams M dsps D} for the given hierarchical cell.
    # Must be called after "open_run synth_1" so a synthesised netlist is loaded.
    # BRAM count is expressed in RAMB36 equivalents (2 x RAMB18 = 1 RAMB36).

    if {[get_cells -quiet $cell_path] eq ""} {
        puts "WARNING: query_cell_resources: cell '$cell_path' not found — returning zeros."
        return [dict create luts 0 brams 0 dsps 0]
    }

    set report [report_utilization -cells [get_cells $cell_path] -return_string]

    set luts   0
    set ramb36 0
    set ramb18 0
    set dsps   0

    foreach line [split $report "\n"] {
        set line [string trim $line]
        # Match table rows: "| <Name> | <Used> | ..."
        # LUT variants
        if {[regexp {^\|\s+Slice LUTs\s*\*?\s*\|\s+(\d+)\s*\|} $line -> val]} {
            set luts $val
        } elseif {$luts == 0 && [regexp {^\|\s+Total LUTs\s*\|\s+(\d+)\s*\|} $line -> val]} {
            set luts $val
        }
        # RAMB36
        if {[regexp {^\|\s+RAMB36/FIFO\*?\s*\|\s+(\d+)\s*\|} $line -> val]} {
            set ramb36 $val
        } elseif {$ramb36 == 0 && [regexp {^\|\s+RAMB36\b[^|]*\|\s+(\d+)\s*\|} $line -> val]} {
            set ramb36 $val
        }
        # RAMB18
        if {[regexp {^\|\s+RAMB18\b[^|]*\|\s+(\d+)\s*\|} $line -> val]} {
            set ramb18 $val
        }
        # DSP – covers DSP48E2, DSPs, DSP Blocks
        if {[regexp {^\|\s+DSP48E2\s*\|\s+(\d+)\s*\|} $line -> val] ||
            ([regexp {^\|\s+DSPs?\s*\|\s+(\d+)\s*\|} $line -> val] && $dsps == 0) ||
            ([regexp {^\|\s+DSP Blocks\s*\|\s+(\d+)\s*\|} $line -> val] && $dsps == 0)} {
            set dsps $val
        }
    }

    # Convert RAMB18 to RAMB36 equivalents (ceil)
    set brams [expr {$ramb36 + int(ceil(double($ramb18) / 2.0))}]

    puts "  query_cell_resources $cell_path -> LUTs=$luts BRAMs=$brams DSPs=$dsps"
    return [dict create luts $luts brams $brams dsps $dsps]
}

################################################################################
# Auto-Floorplan From Synthesis
# Queries post-synthesis resource usage for a list of hierarchical cells and
# calls generate_multi_dfx_pblocks to size and place the pblocks automatically.
#
# Arguments:
#   cell_names    – Tcl list of hierarchical cell paths (one per PR region)
#   pblock_names  – Tcl list of pblock names, paired 1-to-1 with cell_names
#   lut_margin    – multiplicative overhead factor applied to LUT counts (default 1.2)
#   bram_margin   – multiplicative overhead factor applied to BRAM counts (default 1.2)
#   dsp_margin    – multiplicative overhead factor applied to DSP counts  (default 1.2)
################################################################################

proc auto_floorplan_from_synthesis {cell_names pblock_names {lut_margin 1.2} {bram_margin 1.2} {dsp_margin 1.2}} {
    if {[llength $cell_names] != [llength $pblock_names]} {
        error "auto_floorplan_from_synthesis: cell_names and pblock_names must have the same length"
    }

    puts "================================================================"
    puts " Auto-Floorplanning: querying post-synthesis resource usage"
    puts "================================================================"

    set pblock_configs [list]
    foreach cell_name $cell_names pblock_name $pblock_names {
        set res [query_cell_resources $cell_name]

        set luts  [expr {int(ceil([dict get $res luts]  * $lut_margin))}]
        set brams [expr {int(ceil([dict get $res brams] * $bram_margin))}]
        set dsps  [expr {int(ceil([dict get $res dsps]  * $dsp_margin))}]

        # Ensure at least 1 LUT so the search loop does not trivially succeed
        if {$luts  == 0} { set luts  1 }

        # Accumulate report data (post_synth = raw from synthesis; floorplan_input = after margin)
        global finn_pr_report
        if {![info exists finn_pr_report]} { set finn_pr_report [dict create] }
        dict set finn_pr_report $pblock_name cell $cell_name
        dict set finn_pr_report $pblock_name post_synth $res
        dict set finn_pr_report $pblock_name floorplan_input \
            [dict create luts $luts brams $brams dsps $dsps]

        puts "  $pblock_name <- $cell_name : LUTs=$luts BRAMs=$brams DSPs=$dsps (margin applied)"
        lappend pblock_configs \
            [dict create name $pblock_name cell $cell_name luts $luts brams $brams dsps $dsps]
    }

    generate_multi_dfx_pblocks $pblock_configs
}

################################################################################
# PR Resource Report Generator
# Queries post-implementation utilisation for each PR region, combines it with
# data captured during auto-floorplanning (stored in the global finn_pr_report
# dict), and writes a JSON report covering:
#   1) post_synth       – resource counts from the opened synthesis run
#   2) floorplan_input  – counts after applying the margin factor
#   3) pblock_capacity  – sites available in the allocated pblock
#   4) post_impl        – actual resource counts after implementation
#   5) overhead_pct     – (pblock_capacity - post_impl) / post_impl * 100
#
# Must be called while impl_1 is open (open_run impl_1 -name impl_1) and only
# after auto_floorplan_from_synthesis (which populates the finn_pr_report dict).
################################################################################

proc write_pr_resource_report {cell_names pblock_names {report_file "pr_resource_report.json"}} {
    global finn_pr_report

    if {![info exists finn_pr_report] || [llength [dict keys $finn_pr_report]] == 0} {
        puts "WARNING: write_pr_resource_report: finn_pr_report is empty or unset."
        puts "  This proc must be called after auto_floorplan_from_synthesis."
        return
    }

    puts "================================================================"
    puts " PR Resource Report: querying post-implementation utilisation"
    puts "================================================================"

    set regions_json {}

    foreach cell_name $cell_names pblock_name $pblock_names {
        # 4) Post-implementation utilisation
        set post_impl [query_cell_resources $cell_name]

        # Retrieve data stored by auto_floorplan_from_synthesis / generate_multi_dfx_pblocks
        set post_synth      [dict get $finn_pr_report $pblock_name post_synth]
        set fp_input        [dict get $finn_pr_report $pblock_name floorplan_input]
        set pblock_capacity [dict get $finn_pr_report $pblock_name pblock_capacity]

        # 5) Overhead % = (capacity - actual) / actual * 100 per resource type
        foreach resource {luts brams dsps} {
            set cap  [dict get $pblock_capacity $resource]
            set impl [dict get $post_impl $resource]
            if {$impl > 0} {
                set ovh($resource) [format "%.1f" [expr {($cap - $impl) * 100.0 / $impl}]]
            } else {
                set ovh($resource) "null"
            }
        }

        # Build the JSON object for this region
        set rj "    \"$pblock_name\": {"
        append rj "\n      \"cell\": \"$cell_name\","
        append rj "\n      \"post_synth\": {"
        append rj "\"luts\": [dict get $post_synth luts], "
        append rj "\"brams\": [dict get $post_synth brams], "
        append rj "\"dsps\": [dict get $post_synth dsps]},"
        append rj "\n      \"floorplan_input\": {"
        append rj "\"luts\": [dict get $fp_input luts], "
        append rj "\"brams\": [dict get $fp_input brams], "
        append rj "\"dsps\": [dict get $fp_input dsps]},"
        append rj "\n      \"pblock_capacity\": {"
        append rj "\"luts\": [dict get $pblock_capacity luts], "
        append rj "\"brams\": [dict get $pblock_capacity brams], "
        append rj "\"dsps\": [dict get $pblock_capacity dsps]},"
        append rj "\n      \"post_impl\": {"
        append rj "\"luts\": [dict get $post_impl luts], "
        append rj "\"brams\": [dict get $post_impl brams], "
        append rj "\"dsps\": [dict get $post_impl dsps]},"
        append rj "\n      \"overhead_pct\": {"
        append rj "\"luts\": $ovh(luts), "
        append rj "\"brams\": $ovh(brams), "
        append rj "\"dsps\": $ovh(dsps)}"
        append rj "\n    }"

        lappend regions_json $rj

        puts "  $pblock_name post_impl: LUTs=[dict get $post_impl luts] BRAMs=[dict get $post_impl brams] DSPs=[dict get $post_impl dsps]"
        puts "    overhead: LUTs=$ovh(luts)% BRAMs=$ovh(brams)% DSPs=$ovh(dsps)%"
    }

    set json "{\n  \"pr_regions\": {\n"
    append json [join $regions_json ",\n"]
    append json "\n  }\n}"

    set fh [open $report_file w]
    puts $fh $json
    close $fh

    puts "================================================================"
    puts " PR Resource Report written to: $report_file"
    puts "================================================================"
}
