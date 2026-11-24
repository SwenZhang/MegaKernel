for smem in 0 16 32; do
  echo  
  ./rmsnorm 4096 256 0 "$smem" 1000
done
