The source files have been taken from [https://travis-ci.org/cgcostume/cgcostume.github.io](https://travis-ci.org/cgcostume/cgcostume.github.io). and modified according to my needs.

Portfolio optimized for researchers and those who strive for a minimal, file-based "content management" (**Datafolio**).
The complete site's content is based on a json/xml file per section (e.g., ```contact.json```, ```projects.json```, ```publications.json```, ```talks.json```, and ```teaching.json```) as well as the pages ```_config.yml``` information. 

#### Examples

* [Javier Saez](http://jsaezgallego.com)

* [Daniel Limberger](http://www.daniellimberger.de) (adapted)

* [Amir Semmo](http://asemmo.github.io/) (adapted, older revision of Datafolio)

#### Features

* responsive single-page using [Bootstrap 4](http://v4-alpha.getbootstrap.com/)
* multi-language support
* sections for publications, projects, talks/keynotes, teaching, contact, and more
* section contents loaded from json data-files (_data)
* unique, distinguishable layouts per section
* dynamic integration of [Flickr photo sets](https://www.flickr.com/services/api/) (with basic caching)
* php and javascript free contact form using [Formspree](http://formspree.io/)
* optimized for [GitHub Pages](https://pages.github.com/) deployment (uses no unsupported plugins)
* minimizes html and css (currently using compress layout method)
* takes advantage of [jsDelivr](https://www.jsdelivr.com/) and [Google Fonts](https://www.google.com/fonts)
* easy BibTeX provisioning (show, select, and copy to clipboard, as well as download .bib)
* valid html5 output (nearly-valid css, due to some issues in bootstrap)
* responsive navigation (with scrollspy) comprising a top-page link (author or icon), section links (nav-links or dropdown-items), and a language toggle for all used languages
* support for vCard via file and QR Code
* basic [Travis CI](https://travis-ci.org/) integration

#### Dependencies

Datafolio uses [Jekyll](http://jekyllrb.com/), [Bootstrap 4](http://v4-alpha.getbootstrap.com/), [Blueimp Gallery](https://github.com/blueimp/Bootstrap-Image-Gallery), can access the [Flickr API](https://www.flickr.com/services/api/) and relies on [jsDelivr](https://www.jsdelivr.com/), [Google Fonts](https://www.google.com/fonts), and [Formspree](http://formspree.io/).


#### Building the Website (any platform)

* Install [jekyll](https://jekyllrb.com/docs/quickstart/) and follow the given instructions

* Auto rebuilding (on any change) and watching the website (stored in '_site') can be done by running (requires only jekyll)
```
jekyll serve
```
With the server running, the website should be available at http://localhost:4000.
