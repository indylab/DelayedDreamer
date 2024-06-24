standard:
    dmc_proprio done
    mujoco_proprio done
    dmc_vision done

delayed:
    dmc_proprio done
    mujoco_proprio done

    dmc_vision
        latent
            dmc_acrobot_swingup
            dmc_finger_spin
        memoryless
            dmc_cartpole_balance
            dmc_acrobot_swingup
            dmc_finger_spin
        extended done


copy and move:

find . -name "mujoco_Walker2d-v4" -print | cpio -pvdumB /disk00/indylab/karamzaa
find . -name 'checkpoint.ckpt' -exec rm -rf {} +