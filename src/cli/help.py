"""
Display formatted help for available tasks with colors."""

from src.enums.enums import Colors


def display_help():
    """Display help in a professional format with colors."""
    c = Colors

    help_text = f"""
{c.CYAN}{c.BOLD}Description:{c.ENDC}
  Brain Tumor Detection ML Project - Task automation and workflow management.

{c.CYAN}{c.BOLD}Usage:{c.ENDC}
  poetry run task <task_name>

{c.CYAN}{c.BOLD}Available Tasks:{c.ENDC}
  
  {c.GREEN}{c.BOLD}Training and Phases:{c.ENDC}
    {c.YELLOW}dev{c.ENDC}                    Run training (Phase 1)
    {c.YELLOW}t1{c.ENDC}                     Run Phase 1: Training
    {c.YELLOW}t2{c.ENDC}                     Run Phase 2: Basic calibration and evaluation
    {c.YELLOW}t3{c.ENDC}                     Run Phase 3: MLP with uncertainty
    {c.YELLOW}t4{c.ENDC}                     Run Phase 4: CNN with temperature scaling
    {c.YELLOW}t5{c.ENDC}                     Run Phase 5: Decision engine and triage

  {c.GREEN}{c.BOLD}Accuracy Improvements:{c.ENDC}
    {c.YELLOW}boost{c.ENDC}                  Run accuracy boost optimization
    {c.YELLOW}base64{c.ENDC}                 Run simplified accuracy boost
    {c.YELLOW}analyze{c.ENDC}                Analyze accuracy issues

  {c.GREEN}{c.BOLD}Web Interface:{c.ENDC}
    {c.YELLOW}dashboard{c.ENDC}              Launch Flask dashboard server (http://localhost:5000)

  {c.GREEN}{c.BOLD}Code Quality:{c.ENDC}
    {c.YELLOW}lint{c.ENDC}                   Run Black formatter + Pylint (complete check)
    {c.YELLOW}black{c.ENDC}                  Format code with Black
    {c.YELLOW}pylint{c.ENDC}                 Check code with Pylint
    {c.YELLOW}radon{c.ENDC}                  Analyze code complexity
    {c.YELLOW}radon_raw{c.ENDC}              Raw metrics analysis
    {c.YELLOW}radon_hal{c.ENDC}              Halstead metrics analysis
    {c.YELLOW}vulture{c.ENDC}                Find dead code

{c.CYAN}{c.BOLD}Options:{c.ENDC}
  -h, --help               Display this help message
  --version                Show version information

{c.CYAN}{c.BOLD}Examples:{c.ENDC}
  {c.BOLD}poetry run task dev{c.ENDC}      {c.RED}# Train the model{c.ENDC}
  {c.BOLD}poetry run task dashboard{c.ENDC} {c.RED}# Start the web interface{c.ENDC}
  {c.BOLD}poetry run task lint{c.ENDC}     {c.RED}# Check code quality{c.ENDC}

{c.CYAN}{c.BOLD}Help:{c.ENDC}
  For detailed information about environment setup, see README.md
  For task-specific help, check the corresponding script file.
"""
    print(help_text)


if __name__ == "__main__":
    display_help()
