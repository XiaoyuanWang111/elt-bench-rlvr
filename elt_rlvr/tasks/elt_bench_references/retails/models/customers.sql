with order_stats as (
    select
        o_custkey,
        count(*) as num_orders,
        avg(o_totalprice) as average_total_price_per_order
    from orders
    group by o_custkey
),
ranked_orders as (
    select
        o_custkey,
        o_orderdate,
        row_number() over (partition by o_custkey order by o_totalprice desc, o_orderdate asc) as price_rank
    from orders
),
overall as (
    select avg(c_acctbal) as average_balance
    from customer
)
select
    c.c_custkey,
    c.c_nationkey,
    c.c_name,
    coalesce(s.num_orders, 0) as num_orders,
    s.average_total_price_per_order,
    case when c.c_acctbal > o.average_balance then 1 else 0 end as has_above_average_account_balance,
    r.o_orderdate as order_date_highest_total_price,
    case when c.c_acctbal < 4000 and upper(n.n_name) = 'UNITED STATES' then 1 else 0 end
        as balance_lower_than_4000_and_in_us
from customer c
cross join overall o
left join order_stats s on s.o_custkey = c.c_custkey
left join ranked_orders r on r.o_custkey = c.c_custkey and r.price_rank = 1
left join nation n on n.n_nationkey = c.c_nationkey
