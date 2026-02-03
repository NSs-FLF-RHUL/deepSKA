# Contributor Guidelines

## Development Workflow

The deepSKA development workflow follows [github-flow](https://scottchacon.com/2011/08/31/github-flow):

- The `main` branch is kept clean and is always functional.  
  Any code development branches off (with a `git branch`), and eventually merges back into, the `main` branch.
  Core developers have write access, allowing them to open new branches in this repository for their development.
  External contributors should [use forks](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) instead.
  These instructions assume repo access, but their workflow applies to contributions via forks as well.
- Any code development you would like to contribute has to relate to a [GitHub issue](https://github.com/NSs-FLF-RHUL/deepSKA/issues).  
  If an issue does not exist for the feature or bug-fix you want to work on, please open a new issue, using the appropriate template.
- Clone this repo locally to work on your code changes.
- Create a branch off `main`, descriptively named, ideally starting with the issue number, and switch to it:

```bash
$ pwd
/path/to/your/clone/of/deepSKA
$ git status
On branch main

Nothing to commit
$ git checkout -b name-of-my-branch
Switched to a new branch <name-of-my-branch>
```

or

```bash
git switch -c name-of-my-branch
Switched to a new branch <name-of-my-branch>
```

<!-- **Now that you are ready to code, please read the [developers intro](developerIntroduction.md) and [coding standard](codingStandard.md)!** -->

- Commit to that branch locally and regularly, using `git add` and `git commit`.  
  Please use [descriptive commit messages](https://chris.beams.io/git-commit).
- Push your work regularly to the same named branch on the repo via `git push` (you might need the full `git push -u origin name-of-your-branch` the first time you do this).  
  Note that you can continue to add commits to your branch after pushing to GitHub - just run `git push` again to send these new commits to GitHub too.
- If you need feedback or you think your branch is ready for merging, navigate back to the repository on GitHub and use the pull requests (PR) tab to open a PR, targeting `main`, with the changes from your branch.  
  In order to connect your PR with the issue it is resolving, include the text `Fixes #<YOUR-ISSUE-NUMBER>` (e.g. `Fixes #15`) in the PR description.
  If you are not quite ready to merge yet but you would like some feedback or help, open a draft pull request instead.
  You can later convert that to a non-draft PR when you are ready.
- Request a code review from the PR menu on the right.  
  You can request review from individual people, or from a team (e.g. `@NSs-FLF-RHUL/rse-team` will forward your requests to all the RSEs in the project).
- Once your PR has been [reviewed and approved](#conditions-for-merging-a-pr), and all the automated tests have passed, you can merge it into `main` and delete the branch from GitHub.  
  We use [squash merges](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-commits), but your individual commits will still be accessible on GitHub in the (now closed) PR.
  At this point, "merge conflicts" may arise if others have merged their PRs into `main` in the meantime, which you will have to [resolve](#resolving-merge-conflicts).
- After merging, remember to bring your local copy of the repo up to date before creating a new branch for your next development task:

```bash
git checkout main
git pull
```

## Conditions for Merging a PR

The pull request template will guide you through the requirements to get your changes approved by the maintainers.
For reference, those are

- If adding a new function or class, add appropriate tests.
- All tests, existing or new, should pass.
  You should strive to run the tests locally before you open the PR, in order to catch errors early, but the Github Actions automation on the deepSKA repo will run them automatically as well when you open a PR.
- Update the documentation and make sure it builds and looks right.
- Add examples and/or tutorials if you added more substantial functionality.
- One approving code review by one of the requested reviewers.

The code review step is a crucial one, to guarantee the quality of code contributions from the community.
It is an iterative process, and you will have to address the reviewer's comments and any concerns.
Keep in mind that those comments are given in good faith and not as judgement on anyone's coding ability, and are meant to support our community of developers in creating the best software we can, for all of us to use.
Seasoned developers would testify to how much they have learned and improved in their work by receiving reviews on their codes.

## Resolving Merge Conflicts

### Resolving Conflicts via GitHub's Editor

GitHub has a limited online editor that can resolve simple conflicts, where one can select which of the conflicting changes to accept (or to accept a combination of the two).
The editor can be accessed by clicking the "resolve conflicts" box at the bottom of the PR page near the "status checks" box.

The editor will open the files that contain the conflicts and highlight the relevant text:

```text
<<<<<<< name-of-your-branch
some-code...
=======
some-different-code...
>>>>>>> main
```

Everything above the `======` is the changes that are on your branch.
Everything below it are the things on `main` that have changed since you made your branch.
You need to edit these lines so they are consistent and include the changes you want to merge.
You will also need to delete the placeholder lines beginning with `<<<<<<<`, `=======`, and `>>>>>>>`.

Once you have resolved the conflict, hit the "mark as resolved" button in the GitHub editor and then the "commit changes" button to add another commit to your branch with the resolution.
This PR will then auto-update to reflect the resolution.

### Resolving Conflicts via `git` on the Terminal

You can preemt conflicts by bringing your branch up to date with `main` before attempting a merge.

- In your local repo, use

```bash
git checkout main
git pull
```

to bring your local copy of `main` up to date with the upstream.

- Then use

```bash
git checkout name-of-your-branch
git merge main
```

to merge any changes that happened in `main` since you branched off into your branch.
If there are conflicts, you will get some warnings from `git` in the terminal about incompatible changes.

- Open the conflicted files in your favourite editor and fix the conflicts the same way as described above for the Github editor.

- Once you have resolved the conflicts, commit the file as normal with `git add` and `git commit`.
  Finally, `git push` again, and the PR will auto-update to reflect the resolution.

  _Note for experienced git users: `git rebase` should be avoided for bringing your branch up to date with `main` if multiple people might be contributing to a branch (use `git merge` instead as above)._
