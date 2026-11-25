# Incident loss ownership and financial views

A setup freezes each incident law with its physical origins, members and loss components. Shared losses and component identity come from explicit component IDs, never from matching provider names, marginal distributions or simulation row numbers.

## Loss and view definitions

"Physical" means an identified real-world loss and its ownership. It includes financial theft, business interruption and fines as well as bodily injury and property damage. Each distinct component enters unique gross once. Harm to different parties can be distinct, but compensation or insurance reimbursement of a recorded loss is a flow against that loss, not another gross component.

| View | Meaning |
|---|---|
| `gross` | All distinct modeled components, including fines and other transfers, plus each contextual node's annual lab business loss in a year with at least one incident |
| `welfare` | Resource costs, monetary injury valuations and appropriated assets; ordinary transfers, including fines, have weight zero |
| `net_resource_cost` | As `welfare`, but appropriated transfers also have weight zero |
| `provider_contribution` | Each component's whole amount counted once for every provider that contributed to it, so contributions overlap across providers |
| `lab_loss::<lab_id>` | Incident-linked ultimate economic costs and liabilities of the named lab or consolidated group, before recoveries |
| `scenario::<id>::<measure>` | The `gross`, `welfare` or `net_resource_cost` view restricted to one scenario. The measure is required: a plan that requests `scenario::<id>` alone is refused |
| `financial` | Payments under authored contracts after liability, coverage and policy terms |
| `oep_gross` | The largest unique physical initiating-origin loss in the year |
| `oep_financial` | The largest single policy-occurrence payment in the year, summed across contracts that share the occurrence ID, after occurrence retentions and limits and before annual retentions and limits |

`welfare` counts appropriated principal, such as stolen funds, before any recovery, and `net_resource_cost` gives it weight zero; the difference is a convention. A provider's own resources can be distinct costs, but a provider's reimbursement of an external loss is not new harm.

A scenario's component composition must describe the same occurrence population as its loss law. Component shares split the capped shared total, and additive components are amounts outside it. A funds-only total does not contain response costs, and a population of paid claims does not cover all injuries.

Business interruption is lost operating contribution after avoided costs, plus distinct incremental response resources. For example, USD 100 of foregone revenue, USD 60 of avoided cost and USD 10 of additional resources make a USD 50 loss to the affected business. A capitalized value of the same cash flows is not another loss. If replacement production fully offsets the lost contribution within the welfare standing, that transferred margin cancels and the incremental resources remain.

Contribution (`contributor_ids`) associates every contributing provider with the whole of a component. Attribution (`attribution_weights`, which sum to one) divides it. Neither is a legal allocation or a recovery. Where a component's composition is missing, the views that need it are unavailable even when unique gross is identified. An alternative ownership assumption needs complete compatible laws and its own results.

## Lab economic loss

`lab_loss::<lab_id>` and `welfare` read the same sampled components in the same years. The lab view counts the lab's authored own costs, business interruption, third-party liabilities, fines and defense, before insurance or other recoveries. A consolidated group is one declared lab entity with an explicitly authored allocation, not a sum of overlapping lab views. Pooled views need one currency and monetary basis.

Contextual setups give every provider-borne component lab weight one and every other component the elicited lab compensation share. They elicit the lab's business loss as one annual law per node, drawn once in each simulated year whose count is at least one (`ACC-LAB-ANNUAL-BUSINESS-LOSS`). Each episode's lab loss and each node's annual business loss are capped at the stipulated frontier-lab value of USD 1 trillion, the same for every lab (`SEV-LAB-CAP`). Where the tail-class endpoint is lower, it is the cap. A node's lab loss in a year is therefore:

```text
Σ_episodes min(provider-borne + compensation, cap) + 1{N ≥ 1} · min(annual business loss, cap)
```

Limited liability bounds what the lab can bear. The excess falls on creditors, counterparties or victims and leaves `welfare` unchanged, because `welfare` counts the resource loss whoever bears it. Authored setups can add an episode-level aggregate lab law (`AggregateLabLossLaw`); one episode's lab loss is then `min(Σ amount × lab weight + aggregate amount, episode_cap)`.

The annual business loss is not part of any component or of any incident's gross severity, so admission and every per-incident law are unchanged. Besides the lab's own views, it adds to `gross` and `scenario::<id>::gross`: gross counts every party's loss, transfers included, and the lab is a party. It does not enter `welfare`, `net_resource_cost`, their scenario and source-status views or `oep_gross`. Lost revenue and franchise value are largely transfers to competitors and customers, not resource costs to society, and the annual amount is not an occurrence. Because gross counts it, a sample whose answers declare the lab's loss unavailable is left out of the gross views as it is of the lab-loss views (`POOL-EXCLUDED-SAMPLES`).

Each component's `lab_loss_weights` map lab IDs to fractions of its amount; a missing or null map leaves lab loss unavailable, never zero. The weight is the component's whole lab charge, counted once: the reducer adds no bearer attribution, legal obligation, payment or insurance term to it. A fine of 20 with nature `transfer` and lab weight 1 adds 20 to lab loss and zero to `welfare`. Recovering it does not reduce this pre-recovery view, and insurance payments appear only in `financial`.

## Financial routing

Financial views need authored contracts, which only hand-authored setups carry (`src/catmodel/contracts/`).

- A contract pays a component only through an explicit chain: a policy membership naming the contract and component, the insured entity's legal obligation for it, and a payment fraction no larger than that obligation. Coverage cannot create an obligation.
- Financial totals sum the contracts' exposures without coordination (`SUM_UNCOORDINATED_CONTRACT_EXPOSURE`), so the sum can exceed unique gross.
- A contract has no coverage trigger of its own. It responds to the losses the origin-year ultimate basis assigns to the study year, like an occurrence trigger with no reporting limit ([`FIN-TRIGGER`](../assumptions.md#fin-trigger)).
- Retentions and limits are fixed amounts in the population's currency and monetary basis. Claim inflation before settlement, which raises excess-layer payments faster than ground-up loss, is not modeled ([`FIN-NOMINAL-TERMS`](../assumptions.md#fin-nominal-terms)).
- An unbounded outside-limit expense can leave the financial tail unbounded. A new occurrence transform cannot be applied to stored annual arrays that did not keep the occurrence grouping. Treaty identity, hours clauses, reinstatements and inuring order are not supported.
