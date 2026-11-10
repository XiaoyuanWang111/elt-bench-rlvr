with reviews as (
    select
        app,
        sum(case when sentiment = 'Negative' then 1 else 0 end) as num_negative,
        sum(case when sentiment = 'Positive' then 1 else 0 end) as num_positive,
        sum(case when sentiment = 'Neutral' then 1 else 0 end) as num_neutral,
        avg(sentiment_subjectivity) as avg_subjectivity,
        sum(case when translated_review is null or translated_review = 'nan' then 1 else 0 end) as num_nan_or_null
    from user_reviews
    group by app
)
select
    p.app as app_name,
    coalesce(r.num_negative, 0) as num_negative_comments,
    r.avg_subjectivity as avg_sentiment_subjectivity_score,
    coalesce(r.num_nan_or_null, 0) as num_nan_or_null_comment_reviews,
    100.0 * r.num_positive / nullif(r.num_negative, 0) as per_ratio_between_positive_and_negative_setiments,
    coalesce(r.num_neutral, 0) as num_reviews_with_neutral_attitude
from (select distinct app from playstore) p
left join reviews r on r.app = p.app
