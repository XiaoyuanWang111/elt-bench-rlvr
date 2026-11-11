with supplier_stats as (
    select
        s_nationkey,
        sum(case when s_acctbal < 0 then 1 else 0 end) as num_in_debt,
        count(*) as num_suppliers
    from supplier
    group by s_nationkey
),
customer_stats as (
    select
        c_nationkey,
        count(*) as num_customers,
        avg(c_acctbal) as average_balance
    from customer
    group by c_nationkey
),
order_totals as (
    select
        c.c_nationkey,
        sum(o.o_totalprice) as total_price
    from orders o
    join customer c on c.c_custkey = o.o_custkey
    group by c.c_nationkey
),
overall as (
    select avg(c_acctbal) as average_balance
    from customer
)
select
    n.n_nationkey,
    n.n_name,
    n.n_regionkey,
    coalesce(s.num_in_debt, 0) as num_suppliers_in_debt,
    coalesce(cs.num_customers, 0) as num_customers,
    case when cs.average_balance > o.average_balance then 1 else 0 end as balance_larger_thant_avg_balance,
    coalesce(t.total_price, 0) as total_price_of_all_orders,
    100 * cast(s.num_in_debt as double) / s.num_suppliers as per_indebted_suppliers
from nation n
cross join overall o
left join supplier_stats s on s.s_nationkey = n.n_nationkey
left join customer_stats cs on cs.c_nationkey = n.n_nationkey
left join order_totals t on t.c_nationkey = n.n_nationkey
