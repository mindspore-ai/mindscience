#!/bin/bash
#
# Generate VASP input files
#
# Usage:
#   bash generate_vasp_inputs.sh [--auto]
#
#   --auto: Auto mode, use preset options to generate input files (no interaction)
#
# Dependency: qvasp
# Install: Visit https://qvasp.com/Installation.html

POSCAR="POSCAR"

# Check if qvasp is installed
check_qvasp() {
    if ! command -v qvasp &> /dev/null; then
        echo "Error: qvasp not installed or not in PATH"
        echo "Please visit https://qvasp.com/Installation.html to install"
        exit 1
    fi
}

# Check if POSCAR exists
check_poscar() {
    if [ ! -f "$POSCAR" ]; then
        echo "Error: POSCAR file does not exist"
        echo "Please ensure current directory contains a valid POSCAR file"
        exit 1
    fi
}


# Interactive mode
run_interactive() {
    echo "Using qvasp to generate VASP input files..."
    echo ""
    echo "Please select what to generate:"
    echo "  1) qvasp -k density  (Generate KPOINTS: automatic k-point mesh)"
    echo "  2) qvasp -kline       (Generate KPOINTS: band path)"
    echo "  3) qvasp -relax       (Generate INCAR: structure optimization)"
    echo "  4) qvasp -scf        (Generate INCAR: self-consistent calculation)"
    echo ""
    echo "POTCAR generation example: qvasp -pbe Si (need to specify element)"
    echo ""
    qvasp
}

# Auto mode - generate structure optimization basic input files
run_auto() {
    echo "Auto mode: Using qvasp to generate VASP input files..."
    local exit_code=0

    # 1. Generate structure optimization INCAR
    echo "Generating INCAR (structure optimization)..."
    if ! qvasp -relax; then
        echo "Error: INCAR generation failed"
        exit_code=1
    else
        echo "Done: INCAR generated"
    fi

    # 2. Generate KPOINTS
    echo "Generating KPOINTS..."
    if ! qvasp -k density; then
        echo "Error: KPOINTS generation failed"
        exit_code=1
    else
        echo "Done: KPOINTS generated"
    fi

    return $exit_code
}

# Print generated files
show_output() {
    echo ""
    echo "Generated input files:"
    local all_exist=true
    for f in INCAR KPOINTS POSCAR POTCAR; do
        if [ -f "$f" ]; then
            echo "  [Exists] $f"
        else
            echo "  [Missing] $f"
            all_exist=false
        fi
    done

    if [ "$all_exist" = false ]; then
        echo ""
        echo "Warning: Some files missing, may need manual generation"
    fi
}

# Main logic
main() {
    check_qvasp
    check_poscar

    if [ "$1" = "--auto" ] || [ "$1" = "-a" ]; then
        run_auto
        local auto_exit=$?
        show_output

        if [ $auto_exit -ne 0 ]; then
            echo ""
            echo "Note: Auto mode has issues, please try interactive mode: bash generate_vasp_inputs.sh"
        fi
    else
        run_interactive
        show_output
    fi

    echo ""
    echo "Note: Please modify INCAR file according to your material properties, especially:"
    echo "  - ISPIN (set to 2 for magnetic systems)"
    echo "  - ENCUT (recommended to set to 1.2-1.3 times the maximum value in POTCAR)"
    echo "  - MAGMOM (initial magnetic moments for magnetic atoms)"
}

main "$@"
