
The greedy pick has some limitations, for example is most functions have a name starting with `fn`, but one is named `is_even_or_odd`, the model will still pick `fn` for the majority of the time. From there, `is_even_or_odd` would never be picked.

A fix is to have `generate_from`, score each whole option instead of choosing greedily. Add up the log-probabilities of each option's tokens (plus the closing ") and take the highest.


```py
generate_from(prompt, options, end):
    prompt_ids = encode(prompt)

    best_option, best_score = None, -infinity
    for option in options:
        option_ids = encode(prompt + option + end)[len(prompt_ids):]   # see note 3

        ids = prompt_ids.copy()
        score = 0
        for tok in option_ids:
            logits = get_logits(ids)
            log_probs = log_softmax(logits)   # logits -> log-probabilities
            score += log_probs[tok]           # how likely is this exact token here?
            ids.append(tok)                   # teacher forcing: feed the option's own token

        score /= len(option_ids) # see note 2

        if score > best_score:
            best_option, best_score = option, score

    return best_option
```

1. **Use log_softmax, not raw logits.** Raw logits from different steps aren't on the same scale, so you can't add them. Log-probabilities can be added because they're log P(token | everything before it). `log_softmax(x) = x - logsumexp(x)`, and `math/numpy/torch` all have this.

2. **Long options pay more.** Every extra token adds a negative number, so a sum favors short names. Dividing by the number of tokens (the mean log-prob) removes that bias. Try both on your tests. For function names the sum usually works, because the later tokens of the right name are close to log 0.

3. **Tokenize as one string, then cut.** `encode(prompt + option + end)` and then drop the prompt tokens. Encoding option on its own can split differently at the boundary with ". Check that the prompt tokens really are a prefix. They are here, because the prompt ends with ".

4. **Cost.** It takes one forward pass per token, for every option. With about 3 functions × about 6 tokens that's about 18 passes, compared with about 6 now. To avoid recomputing shared prefixes like fn_, walk the options as a trie and cache the logits for each prefix. That's an easy optimization, but it isn't needed for correctness.

5. **`end` is included** (the closing `"`). That's what lets `a` and `ab` be compared fairly: `a"` has to beat `ab"` on the `"` token.

# @ logs and probabilities

We add their **logs**, which is the same as **multiplying** the probabilities.

The probability of a whole option is the chain rule: each token's probability given everything before it, multiplied together.

```
P('is_even_or_odd"') = P("is")                     # first token
                     × P("_even" | "is")
                     × P("_or"   | "is_even")
                     × P("_odd"  | "is_even_or")
                     × P('"'     | "is_even_or_odd")
```

Taking the log turns that product into a sum:

log P(option) = log P("is") + log P("_even" | …) + … + log P('"' | …)

| option | token probabilities | product |
|---|---|---|
| `fn_greet"` | 0.80 × 0.10 × 0.9 | **0.072** |
| `is_even_or_odd"` | 0.20 × 0.95 × 0.95 × 0.95 × 0.9 | **0.154** |


Greedy compares only the first tokens (0.80 vs 0.20) and is stuck with `fn_greet`.