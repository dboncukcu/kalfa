from rich.text import Text
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, Static, Tree

from kalfa.commands import legos
from kalfa.core import aliases


class LegosView(Horizontal):
    found = {}
    shown = None

    def compose(self):
        with Vertical(id="legos-side"):
            yield Input(placeholder="a word to search; enter or escape goes to the list", id="legos-search")
            yield Tree("legos", id="legos-tree")
        with VerticalScroll(id="legos-scroll"):
            yield Static(id="legos-page", markup=False)

    def on_mount(self):
        self.query_one("#legos-tree", Tree).show_root = False
        self.fill("")

    def on_input_changed(self, event):
        if event.input.id == "legos-search":
            self.fill(event.value)

    def on_input_submitted(self, event):
        if event.input.id == "legos-search":
            self.query_one("#legos-tree", Tree).focus()

    def on_tree_node_highlighted(self, event):
        self.show(event.node)

    def show(self, node):
        current = node is not None and node is self.query_one("#legos-tree", Tree).cursor_node
        data = node.data if current and node is not self.shown else None
        if isinstance(data, aliases.Entry):
            page = legos.entry_renderable(data.name)
        elif isinstance(data, str) and data in self.found:
            page = legos.category_renderable(data, self.found[data])
        else:
            return
        self.shown = node
        self.query_one("#legos-page", Static).update(page)

    def fill(self, word):
        word = word.strip() or None
        found = legos.categories(word)
        self.found, self.shown = dict(found), None
        tree = self.query_one("#legos-tree", Tree)
        tree.clear()
        tree.cursor_line = -1
        for category, entries in found:
            branch = tree.root.add(Text(f"{category} ({len(entries)})", style="bold"), category, expand=bool(word))
            for entry in entries:
                branch.add_leaf(legos.entry_label(entry), entry)
        if not found:
            self.query_one("#legos-page", Static).update(legos.empty_text(word, None))
            return
        tree.move_cursor_to_line(1 if word else 0)
        self.show(tree.cursor_node)
