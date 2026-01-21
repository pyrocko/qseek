import inspect
from typing import Callable, Dict, Union

from nicegui import background_tasks, helpers, ui
from pydantic import ValidationError, create_model
from starlette.routing import compile_path


class RouterFrame(ui.element, component="router_frame.js"): ...


class Router:
    def __init__(self) -> None:
        self.routes: Dict[str, Callable] = {}
        self.content: ui.element = RouterFrame()

    def add(self, path: str):
        def decorator(func: Callable):
            self.routes[path] = func
            return func

        return decorator

    def open(self, target: Union[Callable, str]) -> None:
        builder_args = {}

        if isinstance(target, str):
            for pattern, builder in self.routes.items():
                re_pattern, fmt, convertors = compile_path(pattern)
                match = re_pattern.match(target)

                if match:
                    path = target
                    matched_params = match.groupdict()
                    for key, value in matched_params.items():
                        matched_params[key] = convertors[key].convert(value)

                    bound_args = inspect.signature(builder, eval_str=True)
                    model = create_model(
                        "Params",
                        **{
                            key: (_type.annotation, ...)
                            for key, _type in bound_args.parameters.items()
                        },
                    )
                    try:
                        builder_args = model.model_validate(matched_params).model_dump()
                    except ValidationError as exc:
                        ui.label(str(exc)).classes("font-mono text-red-500")
                        return
                    break
            else:
                ui.label("No match")
                return
        else:
            path = {v: k for k, v in self.routes.items()}[target]
            builder = target

        async def build() -> None:
            with self.content:
                ui.run_javascript(f"""
                    if (window.location.pathname !== "{path}") {{
                        history.pushState({{page: "{path}"}}, "", "{path}");
                    }}
                """)
                try:
                    result = builder(**builder_args)
                except Exception as e:
                    ui.label(str(e)).classes("font-mono text-red-500")
                if helpers.is_coroutine_function(builder):
                    await result

        self.content.clear()
        background_tasks.create(build())

    def frame(self) -> ui.element:
        self.content = RouterFrame().on("open", lambda e: self.open(e.args))
        return self.content
