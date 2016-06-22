for i in {1..1}
do
  echo "shuffle data"
  python shuffle_data.py
  python transferLeNet.py
done
