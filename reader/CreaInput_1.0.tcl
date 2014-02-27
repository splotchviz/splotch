#!/usr/bin/wish

puts "Input hierarchy file name"
set file [gets stdin]


set outfile "$file.reduced"
puts "Opening $file"
puts "Writing $outfile"

set file_to_read [open $file "r"]
set file_to_write [open $outfile "w"]

set filedata [read $file_to_read]
close $file_to_read
set filedata [split $filedata "\n"]
set datanumber [llength $filedata]

set numberofgrids "0"

for {set j 0} {$j < $datanumber} {incr j} {

    set auxvar [lindex $filedata $j]
    set auxvar1 [split $auxvar " "]
    set auxvar2 [lindex $auxvar1 0]
    if {$auxvar2 == "Grid"} {incr numberofgrids}
}

puts $file_to_write $numberofgrids

for {set j 0} {$j < $datanumber} {incr j} {

    set auxvar [lindex $filedata $j]
    set auxvar1 [split $auxvar " "]
    set auxvar2 [lindex $auxvar1 0]

# Be careful !!! Indexes of lindex counts also for blanks

    if {$auxvar2 == "Grid"} {

        set grid [lindex $auxvar1 2]
        #puts $file_to_write $grid

    }

    if {$auxvar2 == "GridStartIndex"} {

        set starti [lindex $auxvar1 5]
        set startj [lindex $auxvar1 6]
        set startk [lindex $auxvar1 7]
        puts $file_to_write $starti
        puts $file_to_write $startj
        puts $file_to_write $startk
        
    }

   
    if {$auxvar2 == "GridEndIndex"} {

        set endi [lindex $auxvar1 7]
        set endj [lindex $auxvar1 8]
        set endk [lindex $auxvar1 9]
        puts $file_to_write $endi
        puts $file_to_write $endj
        puts $file_to_write $endk
        
    }

    if {$auxvar2 == "GridLeftEdge"} {

	set startx [lindex $auxvar1 7]
	set starty [lindex $auxvar1 8]
	set startz [lindex $auxvar1 9]
        puts $file_to_write $startx
        puts $file_to_write $starty
        puts $file_to_write $startz

    } elseif {$auxvar2 == "GridRightEdge"} {

	set endx [lindex $auxvar1 6]
	set endy [lindex $auxvar1 7]
	set endz [lindex $auxvar1 8]
        puts $file_to_write $endx
        puts $file_to_write $endy
        puts $file_to_write $endz

    } elseif {$auxvar2 == "BaryonFileName"} {

	set datafilename_aux [lindex $auxvar1 2]
#remove absolute path
        set datafilename [file tail $datafilename_aux]

        puts $file_to_write $datafilename
    }

}

close $file_to_write
exit
