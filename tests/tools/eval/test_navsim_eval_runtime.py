from tools.eval.navsim_eval_runtime import (
    build_generation_kwargs,
    intersect_tokens_preserve_order,
    token_seed,
)


def test_intersect_tokens_preserves_scene_order():
    scene_tokens = ["scene_b", "scene_a", "scene_c", "scene_d"]
    metric_tokens = {"scene_d", "scene_a", "scene_x"}

    result = intersect_tokens_preserve_order(scene_tokens, metric_tokens)

    assert result == ["scene_a", "scene_d"]


def test_build_generation_kwargs_defaults_to_sampling():
    kwargs = build_generation_kwargs(
        {
            "max_length": 2048,
            "temperature": 0.2,
            "top_k": 0,
            "top_p": 1.0,
        }
    )

    assert kwargs == {
        "max_length": 2048,
        "do_sample": True,
        "temperature": 0.2,
        "top_k": 0,
        "top_p": 1.0,
    }


def test_build_generation_kwargs_honors_explicit_do_sample_false():
    kwargs = build_generation_kwargs(
        {
            "max_length": 2048,
            "do_sample": False,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
        }
    )

    assert kwargs["do_sample"] is False


def test_token_seed_is_stable_per_token():
    seed_a = token_seed(17, "token_a")
    seed_b = token_seed(17, "token_b")

    assert seed_a == token_seed(17, "token_a")
    assert seed_a != seed_b
