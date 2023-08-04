---
layout: post
title: "Simple universal bookmarking"
date: 2023-08-03 9:30 
categories: Software
tags: bash dmenu Linux
---

## Bookmarking troubles
I use multiple internet browsers. 
My default browser is [qutebrowser][1], but for some things that qutebrowser doesn't handle well I switch to [Brave][2].
I am also experimenting with [Nyxt][3], which I started exploring as a possible alternative to qutebrowser.
However, it is not yet stable enough for me to make the switch.
Also, some things, like Netflix, do not work on Nyxt yet and I have no idea if there is a way to fix that.

One minor inconvenience when switching between browsers is that my bookmarks are not synchronized across them. 
I have converted my qutebrowser quickmarks to Nyxt bookmarks multiple times, but that is not something that I would like to keep doing, because it is time-consuming.

## A simple solution
I had a vague memory of Luke Smith talking about a simple solution for universal bookmarking that he uses.
With 'universal bookmarking' I mean one bookmarking system that can be used across different browsers.
Luke Smith discusses this in a [Youtube video][4].
His makes use of lightweight tools, such as [xdotool][7], [xclip][8] and [dmenu][9].
I also found [this repo][5] with a script that expands on Luke's idea a bit.
I played around with this expanded solution and then tweaked it further to have something that I am happy about.

My version, at its core, consists of two scripts. 
The first script is basically the command that inserts bookmarks that Luke shows in his video's (with minor edits). 
Luke binds this command directly to a keybind in his DWM configuration.
I found it more convenient to keep the command as a separate script and to just call the script with a similar keybind.
This allows me to modify the script without having to rebuild DWM.

``` bash
#!/bin/sh

xdotool type $(grep -v '^#' ~/.local/share/bookmarks | dmenu -l 20 -F | cut -d' ' -f1)

```

The script finds all the lines in my bookmarks file (a plain text file) that do not start with a `#` (a comment), pipes these into dmenu, allowing me to select one of the bookmarks recorded in the file, and this then gets typed into whatever text field I have selected at the time (using `xdotool type`).
I include titles and tags with my bookmarks, so dmenu should only output the url address itself, which is what the `cut -d' ' -f1` is for.

See the example of an url, as it is recorded in my bookmarks file, below.
``` txt
https://mynoise.net/noiseMachines.php - MyNoise - Audio
```

I also have a script for creating new bookmarks.
Unlike Luke, I opted for not just bookmarking whatever text is currently selected, but I went for something where you can type or paste in a url and then edit it further (following the example of [the earlier mentioned repo][5]).
I also decided to get rid of the part of the script where you paste in the currently selected text altogether.
I think that makes sense when you typically use browsers with normal url-bars, but qutebrowser and Nyxt don't have one.
With those browsers, it makes more sense to just copy the currently visited url (`yy` in qutebrowser) and then paste it into the dmenu prompt (the default keybind for that in dmenu is `C-Y`).

My version of the script thus opens an empty dmenu prompt where you can paste in the url you want to bookmark and type anything else that you want to include alongside it. 
For example, I will typically type in a title and some tags as shown in the example above.

The script then checks if an entry with the url already exists in the bookmark file, ignoring any titles or tags that might be associated with it.
If the url already exists, no new bookmark will be created.

``` bash
#!/bin/sh

file="$XDG_DATA_HOME/bookmarks"

bookmark=$(:|dmenu -p "edit bookmark:")
if [ -z "$bookmark" ]
then
    notify-send -e "bookmark creation cancelled."
    return 1
else
    bookmarkUrl="${bookmark%% *}"
    if grep -q "^$bookmarkUrl" "$file"
    then
	notify-send -e "already bookmarked."
    else
	notify-send -e "bookmark successfully added as $bookmark."
	echo "$bookmark" >> "$file"
    fi
fi
```
I like this simple solution.
It uses, lightweight tools, it uses a simple text file that we can easily edit to keep the bookmarks, and it uses scripts that we can easily adapt or update if we want to.
Also, the bookmarks can be typed in anywhere; not just url-bars.
For example, I could also use it to include links that I've bookmarked in blog posts.
My bookmarking troubles are over.

[1]: https://qutebrowser.org/
[2]: https://brave.com/
[3]: https://nyxt.atlas.engineer/
[4]: https://www.youtube.com/watch?v=d_11QaTlf1I
[5]: https://github.com/nullf6/dmenu-bookmarks/tree/main
[6]: https://dwm.suckless.org/
[7]: https://github.com/jordansissel/xdotool
[8]: https://github.com/astrand/xclip
[9]: https://tools.suckless.org/dmenu/
