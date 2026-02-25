#!/usr/bin/awk -f
# Parse matmul performance log and extract: M_OUT, M, N, K, vgpr_count, vgpr_spill_count, TFlops

BEGIN {
    OFS = ", "
    print "M_OUT, M, N, K, vgpr_count, vgpr_spill_count, TFlops, TFlops_StdDev"
    next_data_line = 0
    tflops_count = 0
}

# Match the first line with M_OUT=, M=, N=, K=
/^M_OUT=/ {
    # Extract M_OUT, M, N, K values
    # Parse: M_OUT=0, M=1, N=2, K=2
    m_out = ""
    m = ""
    n = ""
    k = ""

    # Extract M_OUT
    if (match($0, /M_OUT=[0-9]+/)) {
        val = substr($0, RSTART+6, RLENGTH-6)
        m_out = val
    }

    # Extract M (make sure we don't match M_OUT)
    if (match($0, /M=[0-9]+/)) {
        val = substr($0, RSTART+2, RLENGTH-2)
        m = val
    }

    # Extract N
    if (match($0, /N=[0-9]+/)) {
        val = substr($0, RSTART+2, RLENGTH-2)
        n = val
    }

    # Extract K
    if (match($0, /K=[0-9]+/)) {
        val = substr($0, RSTART+2, RLENGTH-2)
        k = val
    }

    # Reset other variables for this entry
    vgpr_count = ""
    vgpr_spill_count = ""
    tflops_count = 0
    delete tflops_values
    next_data_line = 0
}

# Track when we see the "Triton (TFLOPS)" header
/Triton \(TFLOPS\)/ {
    next_data_line = 1
    next
}

# Match the data line that comes after "Triton (TFLOPS)" header
# Format: 0  8192.0  8192.0  8192.0       1023.378426       824.779925
# Can have 5 or 6 numbers (integer or decimal), we want the last one
# Collect all values until we see a non-data line
next_data_line == 1 && /^[0-9]/ {
    # Extract the last number (Triton TFLOPS)
    # Split by whitespace and get the last field
    nf = split($0, fields, /[ \t]+/)
    if (nf >= 5) {
        tflops_count++
        tflops_values[tflops_count] = fields[nf]
    }
    next
}

# If we see a non-data line while expecting data, stop collecting
next_data_line == 1 && !/^[0-9]/ {
    next_data_line = 0
}

# Match vgpr_count line
/\.vgpr_count:/ {
    # Extract the number after the colon
    # Format: .vgpr_count:     248
    # Find the first number after the colon
    if (match($0, /:[ \t]*[0-9]+/)) {
        val = substr($0, RSTART+1)
        gsub(/^[ \t]*/, "", val)
        gsub(/[^0-9].*$/, "", val)
        vgpr_count = val
    }
}

# Match vgpr_spill_count line
/\.vgpr_spill_count:/ {
    # Extract the number after the colon
    # Format: .vgpr_spill_count: 3
    if (match($0, /:[ \t]*[0-9]+/)) {
        val = substr($0, RSTART+1)
        gsub(/^[ \t]*/, "", val)
        gsub(/[^0-9].*$/, "", val)
        vgpr_spill_count = val
        
        # Stop collecting data lines
        next_data_line = 0
        
        # Calculate average and standard deviation of TFlops
        tflops_avg = ""
        tflops_stddev = ""
        if (tflops_count > 0) {
            # Calculate mean
            sum = 0
            for (i = 1; i <= tflops_count; i++) {
                sum += tflops_values[i]
            }
            tflops_avg = sum / tflops_count
            
            # Calculate standard deviation
            if (tflops_count > 1) {
                variance_sum = 0
                for (i = 1; i <= tflops_count; i++) {
                    diff = tflops_values[i] - tflops_avg
                    variance_sum += diff * diff
                }
                variance = variance_sum / tflops_count
                tflops_stddev = sqrt(variance)
            } else {
                # If only one value, standard deviation is 0
                tflops_stddev = 0
            }
        }
        
        # When we see vgpr_spill_count, we've reached the end of an entry
        # Output the CSV line if we have all required values
        if (m_out != "" && m != "" && n != "" && k != "" && vgpr_count != "" && vgpr_spill_count != "" && tflops_avg != "") {
            print m_out, m, n, k, vgpr_count, vgpr_spill_count, tflops_avg, tflops_stddev
        }
    }
}

