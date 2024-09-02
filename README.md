# Controlled Dropping - Robot Perception and Action Lab USF

This project is an effort to enable fully autonomous robot grasping with precise control of multiple objects.

![Controlled Dropping - Robot Perception and Action Lab](https://rpal.cse.usf.edu/images/dubi2.png)


## Installation

To get started with the project, follow the steps below:

### 1. Environment Setup

The first step is to configure your environment variables. This is done through the provided Bash script.

1. Open your terminal.
2. Navigate to the project directory.
3. Run the environment setup script by executing the following command:

   ```bash
   scripts/setup_env.sh
   ```

   - If the `rpal.env` file already exists, the script will load the environment variables from it.
   - If the `rpal.env` file does not exist, the script will prompt you to enter the values for the required variables (`CONTROL_DROP_DIR` and `SIM_DIR`). The script will then create the `rpal.env` file with your provided values.


### 2. Set Up Conda Environment and Installing the Module

Next, you'll need to set up a Conda environment with Python 3.10 or higher and install the `control_dropping` module.

1. Create a new Conda environment:

   ```bash
   conda create -n rpal_env python=3.10
   conda activate rpal_env
   ```

2. Navigate to the `control_dropping` directory within the project:

   ```bash
   cd control_dropping
   ```

3. Install the module in editable mode:

   ```bash
   pip install -e .
   ```

### Pretraining - Quickstart
1. Download training data from:
 - [Dynamix Data](https://drive.google.com/file/d/1LOnYNdchxagJXLuk8BTKftJcWpd2rIU6/view?usp=sharing)
 - [Critiq Data](https://drive.google.com/file/d/1cw6r8nK05H7W5JEBsY8f7tcb0lRUrn2q/view?usp=sharing)

2. Setup the correct directory structure:
 - Extract the Dynamix data to `ControlDrop-RPAL/Data_Collection/Time_Dependent_Samples_4/
 - Extract the Critiq data to `ControlDrop-RPAL/Data_Collection/Action_Pred_Time_Dependent_Samples_4/

3. Run `scripts/train_dynamix_critiq.sh`:

```bash
scripts/train_dynamix_critiq.sh
```

## Contributors

This project is the result of the collaborative efforts of several individuals and teams at the Robot Perception and Action Lab at USF. Below are the contributors to the "Controlled Dropping - Robot Perception and Action Lab" project:

### Project Contributors

- **[Yu Sun]** - Supervisor
- **[Jonathan Koch]** - Research Assistant
- **[Tianze Chen]** - Supervisor
- **[Adheesh Sennoy]** - Research Assistant
