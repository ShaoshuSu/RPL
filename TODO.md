# RPL Repo Migration TODO

## Boilerplate / Legal
- [ ] Review and update `LICENSE` (currently from old repo template)
- [ ] Review and update `NOTICE`
- [ ] Review and update `CODE_OF_CONDUCT.md`
- [ ] Review and update `CONTRIBUTING.md`

## Git Setup
- [x] Re-add `thirdparty/holosoma` as a proper git submodule (commit `19d19bf`, main)
- [ ] Initial commit of all migrated code
- [ ] Push to `origin` (`git@github.com:amazon-far/RPL.git`)

## README / Docs
- [ ] Update `README.md` — rename references from `holosoma-extension-rpl` to `RPL`
- [ ] Update clone URL in README setup instructions
- [ ] Update `cd holosoma-extension-rpl` to `cd RPL` in README

## Cleanup
- [ ] Remove `__pycache__` and `*.egg-info` if any slipped through (excluded during copy but verify)
- [ ] Verify `.gitignore` is correct for the new repo
- [ ] Verify `.pre-commit-config.yaml` hooks still work
