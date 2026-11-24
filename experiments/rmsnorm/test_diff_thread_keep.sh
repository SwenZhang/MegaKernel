for keep in 0 1; do
  for t in 128 256 512; do
    echo
    ./rmsnorm 4096 "$t" "$keep" 0 1000
  done
done
