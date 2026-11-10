from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from kalfa.commands import guide
from kalfa.core.errors import KalfaError
from kalfa.tui.jobs import pairs_of, words_of


EMPTY = ("the guide explains the config of the config tab, and its path is empty: write a config path there (key 3), "
         "then come back to this tab or press r here")


class GuideView(Horizontal):
    BINDINGS = [("r", "reload", "reload")]
    DEFAULT_CSS = """
    GuideView { height: 1fr; }
    #guide-sections { width: 24; height: 1fr; }
    #guide-scroll { width: 1fr; height: 1fr; }
    #guide-scroll > Static { height: auto; padding: 0 1; }
    #guide-head, #guide-message { margin-bottom: 1; }
    """

    def __init__(self, **options):
        super().__init__(**options)
        self.sections = {}
        self.picked = None
        self.skip = None
        self.shown = None

    def compose(self):
        yield OptionList(id="guide-sections", markup=False)
        yield VerticalScroll(id="guide-scroll")

    async def on_show(self):
        await self.action_reload()

    def fields(self):
        value = self.app.query_one
        paths = words_of(value("#config-path", Input).value)
        sets = pairs_of(value("#config-sets", Input).value, "--set")
        return paths, sets, pairs_of(value("#config-params", Input).value, "-p")

    async def action_reload(self):
        try:
            paths, sets, params = self.fields()
        except KalfaError as exc:
            await self.show_message(f"error: {exc}")
            return
        if not paths:
            await self.show_message(EMPTY)
            return
        doc = guide.document(paths, sets, params)
        await self.show(doc)

    async def fill(self, widgets, names, shown=None):
        scroll = self.query_one("#guide-scroll", VerticalScroll)
        await scroll.remove_children()
        await scroll.mount_all(widgets)
        listing = self.query_one("#guide-sections", OptionList)
        listing.clear_options()
        listing.add_options([Option(name, id=name) for name in names])
        if self.picked in names:
            listing.highlighted = names.index(self.picked)
        elif names:
            self.skip = names[0]
            listing.highlighted = 0
        if self.picked not in names and shown != self.shown:
            scroll.scroll_home(animate=False)
        self.shown = shown

    async def show_message(self, text):
        self.sections = {}
        await self.fill([Static(text, id="guide-message", markup=False)], [])

    async def show(self, doc):
        self.sections = {section["name"]: Static(guide.section_renderable(section, doc), classes="guide-section")
                         for section in doc["sections"]}
        head = Static(guide.head_renderable(doc), id="guide-head")
        await self.fill([head, *self.sections.values()], list(self.sections), doc["title"])

    def on_option_list_option_highlighted(self, event):
        if event.option_list.id != "guide-sections" or event.option_id not in self.sections:
            return
        if event.option_id == self.skip:
            self.skip = None
            return
        self.skip = None
        self.picked = event.option_id
        self.call_after_refresh(self.scroll_to_section, event.option_id)

    def on_option_list_option_selected(self, event):
        if event.option_list.id == "guide-sections" and event.option_id in self.sections:
            self.picked = event.option_id
            self.call_after_refresh(self.scroll_to_section, event.option_id)

    def scroll_to_section(self, name):
        widget = self.sections.get(name)
        if widget is not None and widget.is_attached:
            self.query_one("#guide-scroll", VerticalScroll).scroll_to_widget(widget, animate=False, top=True)
