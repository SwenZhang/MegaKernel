for m in 1 32 256; do
  for t in 128 256 512; do
    	echo
	  ./rmsnorm "$m" "$t" 0 0 1000
  done
done
