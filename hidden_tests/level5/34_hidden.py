"""Private production shape, stride, mask and dynamic-range cases; never prompt material."""

def get_hidden_inputs():
    from kb_workflows.production import make_inputs
    return [make_inputs(34, case=case) for case in ('hidden_odd', 'hidden_strided', 'hidden_extreme')]
