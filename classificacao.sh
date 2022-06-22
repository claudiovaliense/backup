execute(){
    for dataset in $1; do
        for index_fold in $2; do
            python classificacao.py ${dataset} ${index_fold}
        done
    done
}

#execute 'pang_movie_2L' '0 1 2'
#execute 'reut acm webkb 20ng yelp_review_2L pang_movie_2L vader_nyt_2L sst2' '0 1 2 3 4 5 6 7 8 9'
execute 'pang_movie_2L reut webkb yelp_review_2L' '0 1 2 3 4 5 6 7 8 9'
#execute 'sst2' '0'