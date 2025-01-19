# Github Classroom 

## Github 

Each student should create Github account if you don't already have one. Be sure to include your Harvard email as one of the emails associated with the account, as this will give us access to additional features such as GitHub Copilot


One option to avoid inputting your username and password when pushing to a private repository is to use SSH keys. If you've previously set up SSH keys for Github, you can reuse them by copying your existing key pair to the `~/.ssh` directory. Otherwise, you'll need to generate new SSH keys by following these steps:

1. Launch your terminal
2. Run `ssh-keygen -t rsa` and follow the instructions
3. You have now created your ssh keys in the directory `~/.ssh`
4. Run `cat .ssh/id_rsa.pub` to display your public key
5. Copy your public key and enter it into Github under Settings > SSH Keys



## Getting Started with Github Classroom

Please fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLSevzi4x0q_jwUyWH6euwSLc-B0qOkIFzZiAh1dVVQCnOO3inQ/viewform?usp=sharing) so that we can add your username to this GitHub classroom. 

Please read the [Homework Submission Instructions](../chapter_syllabus/syllabus.md#homework-submission-guidelines) for the required format of the homework submission.


### Setting Up Your Repository

When a GitHub classroom assignment is created, we will share a link with you. Click the link and select "Accept this assignment". This will create a forked repository in your personal GitHub account. Note that each team should create **only one** repository for each homework. You should discuss with your team members to decide who will host the repository and how to collaborate with each other.

Follow these steps to get started:

1. Look for an email with the Github Classroom assignment invitation
2. Open the link and select "Accept this assignment" 
3. Wait for your repository to be created - you'll see a confirmation page
4. Find the repository URL by clicking the green "Code" button
5. In your terminal:
   ```bash
   git clone REPO_URL
   ```
6. Follow the [Git tutorial](../chapter_workflow/git.md) to learn how to use Git and GitHub for your homework projects.

### Submitting Assignments

When starting and submitting assignments:

1. Tag the start:
   ```bash
   git tag assignment1-start
   ```

2. When ready to submit:
   ```bash 
   git tag assignment1-submit
   git push origin master --tags
   ```

Make sure to commit all changes before tagging and pushing.
