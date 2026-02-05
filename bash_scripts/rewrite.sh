# list all the bad emails you want to replace
bad_emails=(
  "mitchweikert@macbookpro.mynetworksettings.com"
  "mweik32@login26.chn.perlmutter.nersc.gov"
  "mweik32@login14.chn.perlmutter.nersc.gov"
  mweik32@login16.chn.perlmutter.nersc.gov
  mweik32@login25.chn.perlmutter.nersc.gov
  mweik32@login29.chn.perlmutter.nersc.gov
  # add more if needed
)

good="mweikert2394@gmail.com"

# join them into a space-separated string for use inside the env-filter
bad_list="$(printf '%s ' "${bad_emails[@]}")"

git filter-branch --env-filter "
for bad in $bad_list; do
  if [ \"\$GIT_AUTHOR_EMAIL\" = \"\$bad\" ] || [ \"\$GIT_COMMITTER_EMAIL\" = \"\$bad\" ]; then
    export GIT_AUTHOR_EMAIL=\"$good\"
    export GIT_COMMITTER_EMAIL=\"$good\"
    break
  fi
done
" -- --all

git push --force-with-lease --all
git push --force-with-lease --tags