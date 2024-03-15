# Git Setup



## Getting started

Before you start, make sure you have git installed on your computer. For Mac and Linux user, git is already installed. You can check the git version (also check if it's installed) by this command:

```
git --version
```
You might want to use a code editor, but terminal also works. I'm using [Visual Studio Code](https://code.visualstudio.com).

Then, do the following steps:
- [ ] Generate an [SSH key pair](https://docs.gitlab.com/ee/user/ssh.html). I'm using ED25519 key (without "-sk"). RSA didn't work for me, but you can try and see if it works for you. It's probably a safer one and you can customize the key length. These are the steps that worked for me:

    1. Go to terminal, make sure you're on your home directory and run this command:
    ```
    ssh-keygen -t ed25519 -C "<comment>"
    ```

    2. Enter the file name when the following message appears
    ```
    Enter file in which to save the key (/home/user/.ssh id_ed25519): <filename>
    ```

    The file should be stored in the .ssh folder automatically. 
    
    3. Make sure the \<filename\>.pub shows up in the .ssh directory.
    4. Open config file (create one if you don't have it), then add the following:
    ```
    Host gitlab.com
        PreferredAuthentications publickey
        IdentityFile ~/.ssh/<filename>
    ```
    5. On User settings, click on SSH Keys (if you can't find it, go to search on the top left, and type SSH Keys) and copy the public key to Key box. Instructions can be found on the [user manual](https://docs.gitlab.com/ee/user/ssh.html).
    6. Verify that you can connect by using this command:
    ```
    ssh -T git@gitlab.com
    ```
    A message "Welcome to Gitlab" should appear if you succeed.

- [ ] Now, you can clone the repository. Create a folder where you want to clone the project. Make sure you are on this folder on terminal, then run the following command:
```
git clone git@gitlab.com:qpl/graybox.git
```
Et voila, you now can start working.


## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/qpl/graybox.git
git branch -M main
git push -uf origin main
```

- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

