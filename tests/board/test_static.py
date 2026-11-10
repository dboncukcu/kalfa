import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

import kalfa.board


STATIC = Path(kalfa.board.__file__).parent / "static"
VUE = "https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.esm-browser.prod.js"
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@"
TABS = ["overview", "curves", "model", "data", "predictions", "plots", "config", "files", "log"]
OWNED = [
    "index.html", "app.js", "api.js", "store.js", "format.js", "plot.js", "board.css",
    "components/table.js", "components/chart.js", "components/badge.js", "components/spark.js",
    "components/progress.js", "components/error.js", "components/loading.js", "components/settings.js",
    "components/diagram.js",
    "views/home.js", "views/runs.js", "views/compare.js", "views/sweep.js", "views/record.js",
]
MODULES = [name for name in OWNED if name.endswith(".js")]
TAB_MODULES = [f"views/record/{tab}.js" for tab in TABS]
ALL_MODULES = MODULES + TAB_MODULES
VUE_NAMES = {
    "computed", "createApp", "defineAsyncComponent", "defineComponent", "h", "markRaw", "nextTick", "onBeforeUnmount",
    "onErrorCaptured", "onMounted", "onUnmounted", "onUpdated", "reactive", "readonly", "ref", "shallowRef", "toRaw",
    "toRef", "toRefs", "watch", "watchEffect",
}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
BUILT_IN_TAGS = {"component", "teleport", "template", "slot", "transition", "keep-alive", "transition-group"}
REGEX_BEFORE = set("(,=:[!&|?{};+-*%<>~^")
KEYWORDS_BEFORE_REGEX = {"return", "typeof", "case", "in", "of", "delete", "void", "throw", "new", "else", "do"}
PAIRS = {"(": ")", "[": "]", "{": "}", "${": "}"}
TEMPLATE_GLOBALS = {
    "true", "false", "null", "undefined", "in", "of", "typeof", "instanceof", "new", "this", "Infinity", "NaN",
    "isFinite", "isNaN", "parseFloat", "parseInt", "decodeURI", "decodeURIComponent", "encodeURI",
    "encodeURIComponent", "Math", "Number", "Date", "Array", "Object", "Boolean", "String", "RegExp", "Map", "Set",
    "JSON", "Intl", "BigInt", "console", "Error",
}
IMPORT = re.compile(r"""\bimport\s+([^;]*?)\s*from\s*["']([^"']+)["']""", re.S)
DYNAMIC_IMPORT = re.compile(r"""\bimport\(\s*["']([^"']+)["']\s*\)""")
TEMPLATE = re.compile(r"\btemplate:\s*(`(?:[^`\\]|\\.)*`|'(?:[^'\\]|\\.)*')", re.S)
IDENTIFIER = re.compile(r"[A-Za-z_$][\w$]*")


def source_of(name):
    return (STATIC / name).read_text(encoding="utf-8")


def skip_quoted(text, start):
    quote = text[start]
    position = start + 1
    while position < len(text):
        char = text[position]
        if char == "\\":
            position += 2
            continue
        if char == "\n":
            raise AssertionError(f"unterminated string at {line_of(text, start)}")
        if char == quote:
            return position + 1
        position += 1
    raise AssertionError(f"unterminated string at {line_of(text, start)}")


def skip_regex(text, start):
    position = start + 1
    in_class = False
    while position < len(text):
        char = text[position]
        if char == "\\":
            position += 2
            continue
        if char == "\n":
            raise AssertionError(f"unterminated regex at {line_of(text, start)}")
        if char == "[":
            in_class = True
        elif char == "]":
            in_class = False
        elif char == "/" and not in_class:
            position += 1
            while position < len(text) and text[position].isalpha():
                position += 1
            return position
        position += 1
    raise AssertionError(f"unterminated regex at {line_of(text, start)}")


def line_of(text, position):
    return f"line {text.count(chr(10), 0, position) + 1}"


def scan(text):
    modes = ["code"]
    brackets = []
    before = ""
    position = 0
    while position < len(text):
        char = text[position]
        if modes[-1] == "template":
            if char == "\\":
                position += 2
            elif char == "`":
                modes.pop()
                before = "a"
                position += 1
            elif text.startswith("${", position):
                modes.append("code")
                brackets.append(("${", position))
                before = "("
                position += 2
            else:
                position += 1
            continue
        if char.isspace():
            position += 1
            continue
        if text.startswith("//", position) or text.startswith("/*", position):
            raise AssertionError(f"comment at {line_of(text, position)}")
        if char in "'\"":
            position = skip_quoted(text, position)
            before = "a"
            continue
        if char == "`":
            modes.append("template")
            position += 1
            continue
        if char == "/":
            if before == "" or before in REGEX_BEFORE or before in KEYWORDS_BEFORE_REGEX:
                position = skip_regex(text, position)
                before = "a"
            else:
                before = "/"
                position += 1
            continue
        if char in "([{":
            brackets.append((char, position))
            before = char
            position += 1
            continue
        if char in ")]}":
            assert brackets, f"unmatched {char} at {line_of(text, position)}"
            opener, where = brackets.pop()
            closing = f"{opener} at {line_of(text, where)} closed by {char} at {line_of(text, position)}"
            assert PAIRS[opener] == char, closing
            if opener == "${":
                modes.pop()
            before = "}" if char == "}" and opener != "${" else "a"
            position += 1
            continue
        if char.isalnum() or char in "_$":
            match = IDENTIFIER.match(text, position) or re.compile(r"[\w.]+").match(text, position)
            word = match.group(0)
            before = word if word in KEYWORDS_BEFORE_REGEX else "a"
            position = match.end()
            continue
        before = char
        position += 1
    assert modes == ["code"], "a template literal is not closed"
    assert not brackets, f"{brackets[-1][0]} at {line_of(text, brackets[-1][1])} is not closed"


def matching(text, start):
    depth = 0
    position = start
    while position < len(text):
        char = text[position]
        if char in "'\"`":
            position = skip_quoted(text, position) if char != "`" else text.index("`", position + 1) + 1
            continue
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth -= 1
            if depth == 0:
                return position
        position += 1
    raise AssertionError("unbalanced object")


def top_level_items(body):
    items = []
    depth = 0
    current = ""
    position = 0
    while position < len(body):
        char = body[position]
        if char in "'\"":
            end = skip_quoted(body, position)
            current += body[position:end]
            position = end
            continue
        if char == "`":
            end = body.index("`", position + 1) + 1
            current += body[position:end]
            position = end
            continue
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth -= 1
        if char == "," and depth == 0:
            items.append(current.strip())
            current = ""
        else:
            current += char
        position += 1
    if current.strip():
        items.append(current.strip())
    return items


def item_name(item):
    item = re.sub(r"^\s*(?:async\s+)?", "", item)
    if item[:1] in ("'", '"'):
        return item[1:skip_quoted(item, 0) - 1]
    match = IDENTIFIER.match(item)
    return match.group(0) if match else item.strip()


def object_keys(text, label):
    pairs = {}
    for found in re.finditer(r"\b" + label + r"\s*:\s*\{", text):
        start = found.end() - 1
        body = text[start + 1:matching(text, start)]
        for item in top_level_items(body):
            key, _, value = item.partition(":")
            name = item_name(key)
            pairs[name] = value.strip() or name
    return pairs


def data_keys(text):
    names = set()
    for found in re.finditer(r"\bdata\s*\(\s*\)\s*\{\s*return\s*\{", text):
        start = found.end() - 1
        body = text[start + 1:matching(text, start)]
        names.update(item_name(item.partition(":")[0]) for item in top_level_items(body))
    return names


def option_names(text):
    names = data_keys(text)
    for label in ("computed", "methods"):
        names.update(object_keys(text, label))
    return names


def setup_returns(text):
    if not re.search(r"\bsetup\s*\(", text):
        return set()
    end = TEMPLATE.search(text).start()
    start = text.rindex("return {", 0, end) + len("return ")
    body = text[start + 1:matching(text, start)]
    return {item.partition(":")[0].strip() for item in top_level_items(body)}


def imports_of(text):
    found = []
    for clause, target in IMPORT.findall(text):
        clause = " ".join(clause.split())
        names = []
        default = None
        namespace = None
        braces = re.search(r"\{([^}]*)\}", clause)
        if braces:
            for item in braces.group(1).split(","):
                item = item.strip()
                if item:
                    original, _, alias = item.partition(" as ")
                    names.append((original.strip(), (alias or original).strip()))
            clause = clause[:braces.start()] + clause[braces.end():]
        star = re.search(r"\*\s+as\s+([\w$]+)", clause)
        if star:
            namespace = star.group(1)
            clause = clause[:star.start()] + clause[star.end():]
        rest = clause.replace(",", " ").split()
        if rest:
            default = rest[0]
        found.append({"target": target, "names": names, "default": default, "namespace": namespace})
    return found


def exports_name(text, name):
    declared = re.search(r"\bexport\s+(?:async\s+)?(?:function\*?|const|let|var|class)\s+" + re.escape(name) + r"\b",
                         text)
    listed = re.search(r"\bexport\s*\{[^}]*\b" + re.escape(name) + r"\b[^}]*\}", text)
    return bool(declared or listed)


def resolve(module, target):
    return ((STATIC / module).parent / target).resolve()


def kebab(name):
    return re.sub(r"(?<!^)([A-Z])", r"-\1", name).lower()


def strip_strings(expression):
    return re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", "''", expression)


def expression_names(expression):
    cleaned = strip_strings(expression)
    names = set()
    for match in IDENTIFIER.finditer(cleaned):
        start, end = match.span()
        prefix = cleaned[:start].rstrip()
        suffix = cleaned[end:].lstrip()
        if prefix.endswith(".") and not prefix.endswith("..."):
            continue
        if suffix.startswith(":") and (prefix.endswith("{") or prefix.endswith(",")):
            continue
        if match.group(0).startswith("$") or match.group(0)[0].isdigit():
            continue
        names.add(match.group(0))
    return names


class TemplateReader(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.problems = []
        self.tags = set()
        self.expressions = []
        self.aliases = set()

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        self.read_attributes(attrs)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.add(tag)
        self.read_attributes(attrs)

    def handle_endtag(self, tag):
        if not self.stack:
            self.problems.append(f"</{tag}> closes nothing")
        elif self.stack[-1] != tag:
            self.problems.append(f"</{tag}> closes <{self.stack[-1]}>")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()

    def handle_data(self, data):
        if data.count("{{") != data.count("}}"):
            self.problems.append(f"unbalanced mustache in {data.strip()[:60]!r}")
        self.expressions.extend(re.findall(r"\{\{(.*?)\}\}", data, re.S))

    def read_attributes(self, attrs):
        for name, value in attrs:
            if value is None:
                continue
            if name == "v-for":
                aliases, _, source = value.partition(" in ")
                self.aliases.update(IDENTIFIER.findall(aliases))
                self.expressions.append(source)
            elif name.startswith("#") or name.startswith("v-slot"):
                self.aliases.update(IDENTIFIER.findall(value))
            elif name.startswith(":") or name.startswith("@") or name.startswith("v-"):
                self.expressions.append(value)


def templates_of(text):
    found = []
    for literal in TEMPLATE.findall(text):
        body = literal[1:-1]
        if literal[0] == "'":
            body = body.replace("\\'", "'")
        found.append((literal[0], body))
    return found


@pytest.mark.parametrize("name", OWNED)
def test_owned_files_exist(name):
    assert (STATIC / name).is_file()


@pytest.mark.parametrize("name", ALL_MODULES)
def test_modules_scan_cleanly(name):
    scan(source_of(name))


@pytest.mark.parametrize("name", ALL_MODULES)
def test_relative_imports_resolve_and_export_the_names(name):
    text = source_of(name)
    for found in imports_of(text):
        target = found["target"]
        if target.startswith("http"):
            assert target == VUE, f"{name} imports {target}"
            unknown = {original for original, _ in found["names"]} - VUE_NAMES
            assert not unknown and found["default"] is None, f"{name} imports {unknown} from vue"
            continue
        assert target.startswith("./") or target.startswith("../"), f"{name} imports {target}"
        path = resolve(name, target)
        assert path.is_file(), f"{name} imports missing {target}"
        exported = path.read_text(encoding="utf-8")
        for original, _ in found["names"]:
            assert exports_name(exported, original), f"{target} does not export {original} for {name}"
        if found["default"]:
            assert re.search(r"\bexport\s+default\b", exported), f"{target} has no default export for {name}"


def test_record_view_imports_the_nine_tabs():
    text = source_of("views/record.js")
    for tab in TABS:
        assert f'{tab}: lazy(() => import("./record/{tab}.js"))' in text, f"views/record.js does not load {tab}"


@pytest.mark.parametrize("name", TAB_MODULES)
def test_record_tabs_export_a_default_component(name):
    text = source_of(name)
    assert re.search(r"\bexport\s+default\s*\{", text), f"{name} has no default export object"
    for key in ("record", "path", "api", "version"):
        assert key in object_keys(text, "props"), f"{name} does not declare the {key} prop"


@pytest.mark.parametrize("name", ALL_MODULES)
def test_dynamic_imports_resolve_to_default_exports(name):
    text = source_of(name)
    for target in DYNAMIC_IMPORT.findall(text):
        path = resolve(name, target)
        assert path.is_file(), f"{name} loads missing {target}"
        exported = path.read_text(encoding="utf-8")
        assert re.search(r"\bexport\s+default\b", exported), f"{target} has no default export for {name}"


def local_targets(text):
    found = [target for _, target in IMPORT.findall(text)] + DYNAMIC_IMPORT.findall(text)
    return [target for target in found if not target.startswith("http")]


def reached_from(start):
    seen = []
    queue = [start]
    while queue:
        name = queue.pop()
        if name in seen or not (STATIC / name).is_file():
            continue
        seen.append(name)
        for target in local_targets(source_of(name)):
            queue.append(resolve(name, target).relative_to(STATIC.resolve()).as_posix())
    return seen


def test_every_module_is_reached_from_the_app():
    on_disk = {path.relative_to(STATIC).as_posix() for path in STATIC.rglob("*.js")}
    assert on_disk == set(reached_from("app.js")) == set(ALL_MODULES)


@pytest.mark.parametrize("name", [module for module in ALL_MODULES
                                  if (STATIC / module).is_file() and TEMPLATE.search(source_of(module))])
def test_templates_are_balanced_and_use_known_names(name):
    text = source_of(name)
    templates = templates_of(text)
    assert templates, f"{name} has no template"
    imported = set()
    for found in imports_of(text):
        imported.update(alias for _, alias in found["names"])
        imported.update(item for item in (found["default"], found["namespace"]) if item)
    registered = object_keys(text, "components")
    for alias, value in registered.items():
        assert value in imported or re.search(r"\bconst\s+" + re.escape(value) + r"\b", text), f"{name}: {value}"
    known = set(object_keys(text, "props")) | setup_returns(text) | option_names(text) | TEMPLATE_GLOBALS
    for quote, body in templates:
        assert "${" not in body, f"{name} has an interpolation inside its template"
        assert "`" not in body, f"{name} nests a backtick inside its template"
        reader = TemplateReader()
        reader.feed(body)
        reader.close()
        assert not reader.problems, f"{name}: {reader.problems}"
        assert not reader.stack, f"{name}: unclosed {reader.stack}"
        custom = {tag for tag in reader.tags if "-" in tag} - BUILT_IN_TAGS
        assert custom <= {kebab(alias) for alias in registered}, f"{name}: unregistered {custom}"
        used = set()
        for expression in reader.expressions:
            used |= expression_names(expression)
        unknown = used - known - reader.aliases
        assert not unknown, f"{name}: the template uses {sorted(unknown)} which the component does not define"


def test_vue_comes_from_one_url():
    for name in ALL_MODULES:
        for found in imports_of(source_of(name)):
            if "vue" in found["target"] and found["target"].startswith("http"):
                assert found["target"] == VUE


def test_index_loads_plotly_and_the_app_module():
    page = source_of("index.html")
    assert PLOTLY in page
    assert '<script type="module" src="/static/app.js"></script>' in page
    assert 'href="/static/board.css"' in page and 'id="app"' in page


def test_no_storage_besides_local_storage_and_no_comments():
    for name in ALL_MODULES:
        text = source_of(name)
        for word in ("sessionStorage", "indexedDB", "document.cookie"):
            assert word not in text, f"{name} uses {word}"
        if "localStorage" in text:
            assert name in ("store.js", "components/settings.js", "views/record/predictions.js"), \
                f"{name} uses localStorage outside the store"
    assert "/*" not in source_of("board.css")
    assert "<!--" not in source_of("index.html")


def test_app_routes_every_page():
    text = source_of("app.js")
    for page in ("home", "runs", "compare", "sweep", "record"):
        assert f"route.page === '{page}'" in text
    assert 'addEventListener("hashchange"' in text and "createApp(" in text


def test_api_talks_to_the_contract_routes():
    text = source_of("api.js")
    for route in ("/api/record/", "/api/sweep/", "/api/live"):
        assert route in text
    for name in ("getJson", "postJson", "live", "getText", "recordUrl", "fileUrl"):
        assert exports_name(text, name)
    pages = {"views/sweep.js": ("/api/report",), "views/compare.js": ("/api/compare",), "store.js": ("/api/records",)}
    for module, routes in pages.items():
        for route in routes:
            assert route in source_of(module), f"{module} does not ask {route}"


def test_each_histogram_of_the_predictions_page_sends_its_own_bins():
    chart, settings = source_of("components/chart.js"), source_of("components/settings.js")
    assert "binsBox" in object_keys(chart, "props") and ':bins-box="binsBox"' in chart
    assert "binsBox" in object_keys(settings, "props") and 'kind === "histogram" || Boolean(binsBox)' in settings
    api = source_of("api.js")
    assert exports_name(api, "histogramsText") and 'name + ":" + count' in api
    page = source_of("views/record/predictions.js")
    for chart_name, histogram in (("residuals", "residual_hist"), ("histograms", "truth_hist"),
                                  ("scores", "score_hist"), ("norms", "norm_hist"), ("values", "hist")):
        assert f'{chart_name}: "{histogram}"' in page
    assert "bins: this.bins" not in page and "wantedBins" not in page and "params.histograms" in page
    score = re.search(r'<k-chart v-if="scoreData"[^>]*>', page).group(0)
    assert "bins-box" in score and '@settings="onSettings"' in score


def test_dark_theme_follows_the_system_and_the_toggle():
    css = source_of("board.css")
    assert "@media (prefers-color-scheme: dark)" in css
    assert ':root:not([data-theme="light"])' in css and ':root[data-theme="dark"]' in css
    for index in range(1, 9):
        assert css.count(f"--series-{index}:") == 3
    for status in ("running", "stale", "done", "stopped", "failed"):
        assert f".badge.is-{status}" in css


@pytest.mark.parametrize("broken", [
    "const a = [1, 2;",
    "function f() { return (1 + 2; }",
    "const s = \"open;",
    "const t = `<div>${x</div>`;",
    "const a = 1; // note",
    "const o = { a: 1 ];",
])
def test_scan_catches_broken_modules(broken):
    with pytest.raises(AssertionError):
        scan(broken)


def test_scan_accepts_regexes_and_nested_interpolation():
    scan("const r = /[\\\\/]+/g; const t = `a ${f({ b: [1] })} c`; const d = (4) / 2; if (x) { y(); }\n/^#/.test(z);")


def test_template_check_sees_unknown_names_and_bad_tags():
    reader = TemplateReader()
    reader.feed('<div :class="{ on: flag }" @click="go(item)"><k-x v-for="item in items"></k-x></span>')
    reader.close()
    names = set()
    for expression in reader.expressions:
        names |= expression_names(expression)
    assert names == {"flag", "go", "item", "items"} and reader.aliases == {"item"}
    assert reader.problems and "k-x" in reader.tags


def test_setup_returns_and_props_are_read():
    text = ("const C = { props: { a: { type: String, default: 'x' }, b: Number }, setup() { const f = () => { return "
            "{ z: 1 }; }; return { c, d: f, e }; }, template: `<p>{{ c }}</p>` };")
    assert set(object_keys(text, "props")) == {"a", "b"} and setup_returns(text) == {"c", "d", "e"}


def test_option_names_are_read():
    text = ("export default { props: { record: Object }, data() { return { open: false, name: f(a, b) }; }, "
            "computed: { rows() { return { a: 1 }; }, total() { return 2; } }, methods: { bytes, async load() {} }, "
            "components: { 'k-badge': Badge }, template: `<p>{{ rows }}</p>` };")
    assert option_names(text) == {"open", "name", "rows", "total", "bytes", "load"}
    assert set(object_keys(text, "props")) == {"record"} and object_keys(text, "components") == {"k-badge": "Badge"}


def missing_commas(text):
    code = re.sub(r"`(?:[^`\\]|\\.)*`", "``", text)
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    found = []
    for before, line in zip(lines, lines[1:]):
        if re.match(r"^[A-Za-z_$][\w$]*\s*:(?!:)", line) and not before.endswith((",", "{", "(", "[", ":")):
            found.append(line)
    return found


@pytest.mark.parametrize("name", ALL_MODULES)
def test_object_literals_have_commas(name):
    assert not missing_commas(source_of(name)), f"{name}: {missing_commas(source_of(name))}"


def test_missing_comma_is_seen():
    assert missing_commas("const o = {\n  a: 1\n  b: 2,\n};") == ["b: 2,"]
