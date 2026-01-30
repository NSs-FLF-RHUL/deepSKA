# Contributor guidelines

## Development Workflow

The deepSKA development workflow follows [github-flow](https://scottchacon.com/2011/08/31/github-flow):

- The `main` branch is kept clean and is always functional.
  Any code development branches off and eventually merges back into the `main` branch, via branches.
  The core developers have access to open new branches in this repo for their development, whereas external contributors should [use forks](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) instead.
  These instructions assume repo access, but their workflow applies to contributions via forks as well.
- Any code development you would like to contribute has to relate to a [GitHub issue](https://github.com/NSs-FLF-RHUL/deepSKA/issues).
  If one does not exist for the feature or bugfix you want to work on, please open a new issue, using the appropriate template.
- Clone this repo locally to work on your code changes

- Create a branch off `main`, descriptively named, ideally starting with the issue number, and switch to it, e.g.

```bash
git switch -c 15-implement-solver-class
```

<!-- **Now that you are ready to code, please read the [developers intro](developerIntroduction.md) and [coding standard](codingStandard.md)!** -->

- Commit to that branch locally and regularly push your work to the same named branch on the repo.
  Please use [descriptive commit messages](https://chris.beams.io/git-commit).

- If you need feedback or help but your branch is not ready to merge, open a draft pull request (PR) from your branch to `main`.
  Likewise, when you think the branch is ready for merging, open a (normal) PR, or convert your draft one to a normal PR, and request a code review.
  In order to connect your PR with the issue its is resolving, start the description with `Fixes #<YOUR-ISSUE-NUMBER>`, e.g. `Fixes #15`.

- Once your PR has been [reviewed and approved](#conditions-for-merging-a-pr), and all the automated tests have passed, you can merge it into `main` and delete the branch from GitHub.
  At this point, code conflicts may arise if others have merged their PRs into `main` in the meantime.
  You can resolve those while merging, or preemt them by bringing your branch up to date with `main` before attempting a merge.
  Note for experienced git users: if you want to bring your branch up to date with `main`, `git rebase` should be avoided if multiple people might be contributing to a branch (use `git merge` instead).
  We use [squash merges](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-commits), but your individual commits will still be accessible on GitHub in the (now closed) PR.

- Remember to bring your local copy of the repo up to date before creating a new branch for your next development task.

```bash
git checkout main
git pull
```

## Conditions for merging a PR

The pull request template will guide you through the requirements to get your changes approved by the maintainers. For reference, those are

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
