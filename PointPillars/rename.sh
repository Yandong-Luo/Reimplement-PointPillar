index=0
for file in $(ls *.txt | sort); do
    mv "$file" "${index}.txt"
    index=$((index + 1))
done

