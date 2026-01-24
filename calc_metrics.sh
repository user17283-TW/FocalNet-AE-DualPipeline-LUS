for f in ./output/*.json; do
    echo "File [$f]"
    python postp.py "$f"
    echo -e "\n"
done