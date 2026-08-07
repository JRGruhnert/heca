for i in range(len(data)):
    if oracle_done[i] == 1.0:
        # Success of PREVIOUS oracle (last step before switch)
        success = oracle_success[i - 1] == 1.0
        # Direction: compare start vs target
        start = privileged_faucet_0_pos[seg_start]
        target = heca_target_faucet_0_pos[i - 1]
        direction = "open" if target > start else "close"

        if success:
            save_segment(data[seg_start:i], direction)
        seg_start = i
