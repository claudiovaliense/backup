execute(){
    for dataset in $1; do
        for index_fold in $2; do
            python tfidf_bombado.py ${dataset} ${index_fold} ${3}
        done
    done
}

#execute 'webkb' '0 1 2' '10'
execute 'acm yelp_review_2L' '0 1 2' '10'