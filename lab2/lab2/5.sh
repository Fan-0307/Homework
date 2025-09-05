ps ax | awk '/python/ {print $1}' | sort | uniq | wc -l
