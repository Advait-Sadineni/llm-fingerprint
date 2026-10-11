// Paper-style template for the CMPSC 448 midterm report. Compile from the repo root:
//   typst compile --root . report/report.typ report/report.pdf
#let paper(title: "", author: "", affiliation: "", date: "", abstract: [], body) = {
  set page(paper: "us-letter", margin: (x: 0.8in, y: 0.9in), numbering: "1",
    footer: context [#set text(size: 8pt, fill: luma(110)); #h(1fr) #counter(page).display() #h(1fr)])
  set text(font: "New Computer Modern", size: 10pt)
  set par(justify: true, leading: 0.6em)
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => { v(0.9em); text(size: 12pt, weight: "bold", it); v(0.35em) }
  show heading.where(level: 2): it => { v(0.6em); text(size: 10.5pt, weight: "bold", it); v(0.25em) }
  show figure.caption: it => { set text(size: 8.5pt); it }
  show figure.where(kind: table): set figure.caption(position: top)
  show table: set text(size: 8.5pt)
  set table(stroke: (x, y) => if y == 0 { (bottom: 0.6pt + black, top: 0.6pt + black) } else { none }, inset: (x: 5pt, y: 3.5pt))
  show link: set text(fill: rgb("#1c5cab"))
  show raw.where(block: false): set text(size: 8.5pt)

  align(center)[
    #text(size: 17pt, weight: "bold", title)
    #v(0.6em)
    #text(size: 11pt, author) \
    #text(size: 9.5pt, fill: luma(80), affiliation) \
    #text(size: 9.5pt, fill: luma(80), date)
  ]
  v(0.8em)
  block(inset: (x: 2.2em), [
    #set text(size: 9.5pt)
    *Abstract.* #abstract
  ])
  v(0.8em)
  columns(2, gutter: 1.6em, body)
}

#let fig(path, caption, width: 100%) = figure(image(path, width: width), caption: caption)
#let wide(body) = place(top + center, float: true, scope: "parent", block(width: 100%, body))
