---
layout: post
title:  "A better csv parser"
date:   2017-12-21 09:07:00 +0000
categories: Software 
tags: Q-SoPrA Technical C++ Qt

---

## Background
Before I get to the main topic of this post, I should offer a bit of background. I am currently working on an integrated software package for the qualitative study of social processes, called Q-SoPrA (an acronym for **So**cial **Pr**ocess **A**nalysis). The program can be used for data entry, data management, hierarchical coding for attributes (think CAQDAS-style coding), coding for network relationships, coding for linkages between events (see the work of [David Heise][1] for a main source of inspiration in this), abstracting larger events from smaller ones, visualising event graphs, visualising event hierarchies, visualising network graphs, visualising occurrence graphs (similar to [Bi-Dynamic Line Graphs][2]), and basic analysis. I intend to write some technical as well as non-technical posts about Q-SoPrA and its development. The technical posts will deal primarily with what happens on 'the back stage' of the program, that is, discuss the underlying source code. The non-technical posts will focus primarily on Q-SoPrA's 'front stage', discussing its features and how to use them. This post is going to be a technical one. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/welcome_dialog.png" class="img-fluid rounded z-depth-1" zoomable=true center=true %}
    </div>
</div>
<div class="caption">
Welcome dialog of Q-SoPrA.
</div>
 
The main reason for writing the more technical posts is that I have made heavy use of information and advice on writing code that I found online. In fact, I learned almost everything I know about coding from people willing to share their knowledge via the Web (of course, a lot of it was simply learning by doing). Occasionally, I run into difficult problems that I am only able to solve after seeing various posts on websites like Stack Overflow. That being said, the solutions I finally come up with are typically a variant of a particular suggestion that was made online, or a combination of various suggestions. I thought it was a nice idea to give something back, by discussing some solutions for ~~problems~~ challenges I encountered while writing the code for Q-SoPrA, and making those discussion available online, in the hope that someone in search for answers to similar problems might one day stumble upon my posts. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/SO.png" class="img-fluid rounded z-depth-1" zoomable=true center=true %}
    </div>
</div>
 

## What's the problem?
In this post I will discuss something related to the *Data Widget* that I wrote for Q-SoPrA. I divided Q-SoPrA into several widgets that perform the main tasks of the program (the tasks listed in the background discussion above). The Data Widget handles the user's interaction with data. In this case, the data take the form of chronologically ordered *incidents*, which are bracketed, qualitative descriptions of activities. The concept of incidents was introduced during the [Minnesota Innovation Research Program][3], although I work with a slightly different idea of what information should be included in an incident:

1. An indication of the **time of occurrence** (e.g., the hour, the day, the month, the year).
2. A **description** of the activity captured in the incident (created by the analyst).
3. The **source** of data. 
4. (Optionally / if available) A fragment of **'raw' data**, such as a fragment from an interview transcript, or a passage from a document in which the activity is reported.
5. (Optionally) A **comment / memo** written on the incident by the researcher. 
	
The Data Widget allows the analyst to create incidents, and store them in a table. To enter / edit data, the user can use a popup dialog. After saving a new incident, a row is added to the Data Widget table, after which the user can still change the location of the incident (i.e., change its position in the chronological order of all incidents), edit its details, duplicate it, or remove it. All changes are immediately written to an sqlite database. See a screenshot of the Data Widget table below.
	
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/data_widget.png" class="img-fluid rounded z-depth-1" zoomable=true center=true %}
    </div>
</div>
<div class="caption">
The Data widget.
</div>
 
One thing that the user can also do is to import an existing data set that was stored in [csv format][4]. I implemented this future because I have been building data sets before I started using Q-SoPrA myself (I am already using the program while I am building it, which is also a good way to test it). I wanted to have the option to import these existing data sets. The most obvious way to do this (to me, at least) is to store the data sets in csv files, and then import them into Q-SoPrA. **And here is where we finally get to the main topic of this post:** I needed a good way to parse csv files submitted by the user. This in itself is a somewhat trivial task. However, the csv files that I created in the past are a bit 'messy', primarily due to the fact that I typically included text from sources of 'raw data', such as written news items, and other types of documents. I typically just copy and paste this data, and in a csv file this will often lead to the inclusion of embedded line breaks, and embedded double quotes (e.g., when someone is quoted in a news item). 

Most of the relatively simple csv parsers that I have seen will run into problems when trying to read such messy files. After several iterations, I created the set of functions listed in full below. I included a lot of comments in the code to explain what happens at different steps. Most of the code is standard C++ code, but there are also some objects that are based on the Qt library (e.g., QString, QPointer, QMessageBox, QSqlQuery). This is because I used the Qt5 library for the graphical user interface of Q-SoPrA, as well as for interaction with sqlite data sets.

Let's start with the main function, called `importFromCsv()`, (which calls two other customised functions). 

{% highlight c++ %}
/* 
   This function is designed to facilitate importing data from 
   an external csv file. We read each line of data in turn, 
   check whether we should append any other lines due to 
   embedded line breaks, then cut up the resulting line 
   into tokens, and store these in a data vector before writing 
   the data into our sqlite database.

   We also check whether the file has the correct headers.
*/
void MainWindow::importFromCsv() {
  // We ask the user to open a file to import from.
  QString csvName = QFileDialog::getOpenFileName(this, tr("Select csv file"),"",
						 tr("csv files (*.csv)"));
  // We then create an ifstream object that goes through the file.
  std::ifstream file ((csvName.toStdString()).c_str());
  std::vector<std::vector <std::string> > data; // To store our data in.
  bool headerFound = false; // To be used in a condition further below.
  while (file) {
    // The buffer will hold each line of data as we read the file.
    std::string buffer; 
    if (!getline(file, buffer)) break; // We get the current line/

    // We should check and handle any extra line breaks in the file.
    while (checkLineBreaks(buffer) == true) {
      std::string extra;
      getline(file, extra);
      buffer = buffer + "\n\n" + extra;
    }
    // We need an object to keep the separate tokens in a line.
    std::vector<std::string> tokens; 
    // We then split the current line into different tokens.
    splitCsvLine(&tokens, buffer);
    /* 
       If we still need to import the header, let's do that first.
       We immediately check if the correct headers were used.
       If not, we report an error and return, letting the user fix the issue.
    */
    if (headerFound == false) {
      if (tokens[0] != "Timing" && tokens[0] != "timing") {
	QPointer<QMessageBox> errorBox = new QMessageBox(this);
	errorBox->setText(tr("<b>ERROR</b>"));
	errorBox->setInformativeText("Expected \"Timing\" in first column.");
	errorBox->exec();
	delete errorBox;
	return;
      }
      if (tokens[1] != "Description" && tokens[1] != "description") {
	QPointer<QMessageBox> errorBox = new QMessageBox(this);
	errorBox->setText(tr("<b>ERROR</b>"));
	errorBox->setInformativeText("Expected \"Description\" "
				     "in second column.");
	errorBox->exec();
	delete errorBox;
	return;
      }
      if (tokens[2] != "Raw" && tokens[2] != "raw") {
	QPointer<QMessageBox> errorBox = new QMessageBox(this);
	errorBox->setText(tr("<b>ERROR</b>"));
	errorBox->setInformativeText("Expected \"Raw\" in third column.");
	errorBox->exec();
	delete errorBox;
	return;
      }
      if (tokens[3] != "Comments" && tokens[3] != "comments") {
	QPointer<QMessageBox> errorBox = new QMessageBox(this);
	errorBox->setText(tr("<b>ERROR</b>"));
	errorBox->setInformativeText("Expected \"Comments\" in "
				     "fourth column.");
	errorBox->exec();
	delete errorBox;
	return;
      }
      if (tokens[4] != "Source" && tokens[4] != "source") {
	QPointer<QMessageBox> errorBox = new QMessageBox(this);
	errorBox->setText(tr("<b>ERROR</b>"));
	errorBox->setInformativeText("Expected \"Source\" in "
				     "fifth column.");
	errorBox->exec();
	delete errorBox;
	return;
      }
      /* 
	 If we checked all headers and imported the header correctly,
	 we can set the headerFound bool to true, so that this block 
	 is skipped in all subsequent line reads.
      */
      headerFound = true;
    } else { // This is the block that is run after the header was
      already imported.
      // We iterate through the tokens and push them into the row vector.
      std::vector<std::string> row;
      std::vector<std::string>::iterator it; 
      for (it = tokens.begin(); it != tokens.end(); it++) {
	row.push_back(*it);
      }
      // Then we push the row vector into the data vector.
      data.push_back(row);
    }
  }
  /* 
     Writing to sqlite databases is quite slow, so if the 
     csv-file is large, it will take a while. We use a progress bar 
     to report to the user how much we have already imported 
     into the sqlite database.
  */
  loadProgress = new ProgressBar(0, 1, (int)data.size());
  loadProgress->setAttribute(Qt::WA_DeleteOnClose);
  loadProgress->setModal(true);
  loadProgress->show();

  /* 
     We need a pointer to the data widget to determine the 
     number of rows already in the table
  */
  DataWidget *dw = qobject_cast<DataWidget*>(stacked->widget(0));
  dw->updateTable(); // A function to make sure all table rows are read.
  /* 
     Since imported rows are appended to the end of the data base, 
     we can assume that the order variable is the number of 
     existing rows + 1.
  */
  int order = dw->incidentsModel->rowCount() + 1;
  std::vector<std::vector <std::string> >::iterator it;
  for (it = data.begin(); it != data.end(); it++) {
    // We create all the necessary variables and write them to the table.
    std::vector<std::string> currentRow = *it;
    QString timeStamp = QString::fromStdString(currentRow[0]);
    QString description = QString::fromStdString(currentRow[1]);
    QString raw = QString::fromStdString(currentRow[2]);
    QString comment = QString::fromStdString(currentRow[3]);
    QString source = QString::fromStdString(currentRow[4]);
    QSqlQuery *query = new QSqlQuery; // For this we use the QSqlQuery object.
    query->prepare("INSERT INTO incidents "
		   "(ch_order, timestamp, description, raw, comment, "
		   "source, mark) "
		   "VALUES "
		   "(:order, :timestamp, :description, :raw, :comment, "
		   ":source, :mark)");
    query->bindValue(":order", order);
    query->bindValue(":timestamp", timeStamp);
    query->bindValue(":description", description);
    query->bindValue(":raw", raw);
    query->bindValue(":comment", comment);
    query->bindValue(":source", source);
    query->bindValue(":mark", 0);
    query->exec();		   
    order++; // Make sure that we increment the order variable.
    loadProgress->setProgress(order); // Set progress and report
    qApp->processEvents(); // Make sure that the progress is visible
    delete query; // Memory management
  }
  loadProgress->close(); // We can close the progress bar.
  delete loadProgress; // Memory management
  // The imported data won't be visible unless we run the following lines.
  dw->incidentsModel->select();
  dw->updateTable();
  // We should also update the attributes widget at this point. 
  AttributesWidget *aw = qobject_cast<AttributesWidget*>(stacked->widget(1)); 
  aw->retrieveData();
}
{% endhighlight %}

### Handling embedded line breaks

So, what exactly happens here? Well, I won't discuss all the lines of code in detail, but I will discuss the ones most important to the problem at hand (parsing 'messy' csv-files). The basic logic of the function is that we read a csv file line by line, and that we process each line in turn. One challenge that we might encounter while parsing 'messy' csv files is the presence of embedded line breaks. This means that one cell includes multiple lines of data, separated by so-called line breaks. See the screenshot below for an example. In the second row of data (below the header), we see a couple of line breaks. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/embedded_line_breaks.png" class="img-fluid rounded z-depth-1" zoomable=true center=true %}
    </div>
</div>
<div class="caption">
Embedded line breaks.
</div>
 
The line breaks are even more obvious if we open our csv file with a basic text editor. We will see something this:

```text

timing,description,raw,comments,source
1,test1,"Let's make
this

example",,s1
2,test2,What?,,s2
3,test3,Nothing,,s3

```
<details>
<summary>Click here for some additional info on what we see here</summary>
<blockquote>
Looking at the text fragment above actually gives us a pretty good idea of how a csv file is typically structured. One thing to notice is that we see a lot of commas that we did not see in the original screenshot. These commas are used to separate between different columns of data, and are the reason why csv files are often referred to as comma-delimited files (actually, csv stands for <b>C</b>omma <b>S</b>eparated <b>V</b>alue). In the file above, we sometimes see two commas. This simply means that the cell between those commas is empty. We also see that cells with line breaks in them are double quoted, while we did not add double quotes in our original file. Double quotes are often used for text fields if there is something special about them, such as the presence of line breaks, or the presence of embedded commas (commas that do not indicate additional columns, but simply break up sentences into different segments). 
</blockquote>
</details><br>

If we use the C++ `getline()`function to read the lines in this file, then the first line it will read is `"timing","description","raw","comments","source"`, the second line it will read is `1,"test1","Let's make`, the third line `this`, and so on. This is a bit of a problem, because we actually want the second line of data to be something like `1,"test1","Let's make this example",,"s1"`. Fortunately, the [csv standard][5] prescribes that we embed text fields with line breaks in them in double quotes. We can use the number of double quotes on a given line of data as clue about whether or not that line might have line breaks embedded in them. For example, in the second line of data, we find only 1 double quote, which suggest that we might not be finished reading the line (otherwise we should have encountered a matching closing double quote somewhere down the line).

One thing we could do whenever we encounter an unequal number of quotes, is to check if we can perhaps find the missing quotes in the next line(s) of data. If this indeed the case (it should be the case), then we can append these other lines to our current line in order to complete it.

That is what happens in the `while (checkLineBreaks(buffer) == true)` loop. 'checkLineBreaks(QString line)' is an additional function that I wrote, and that returns true if the parser thinks there is an embedded line break in the current line. I show the while loop and the accompanying 'checkLineBreaks(QString line)' function below.

{% highlight c++ %}
while (checkLineBreaks(buffer) == true) {
  std::string extra;
  getline(file, extra);
  buffer = buffer + "\n\n" + extra;
}
	
/* 
   Function to check for line breaks in current line of imported csv file.
   The logic is that a finished line of data will always have an 
   equal number of double quotes ("). If this is not the case, we can assume 
   that another line of data should be appended. 
*/
bool MainWindow::checkLineBreaks(std::string line) {
  bool lineBreak = false; // This is the variable that will be returned.
  for (std::string::size_type i = 0; i != line.length(); i++) {
    /* 
       If we did not find a quote yet, and we find one,
       we set the lineBreak variable to true.
    */
    if (lineBreak == false && line[i] == '"') {
      lineBreak = true;
      /*
	If we then find another quote, 
	we can assume that no embedded line break exists.
      */
    } else if (lineBreak == true && line[i] == '"') {
      lineBreak = false;
    }
  }
  // After reading the line, we return the result.
  return lineBreak;  
}
{% endhighlight %}

The `buffer` object is the original line of data that we are currently reading. If we find a line break in this object, then we create a new object `extra`, put the next line of data in it, and append it to `buffer` (we also include two line breaks to mimic the original line break; why I chose to use two has to do with something that is not important right now). Since we are using a while loop, the program will keep doing this as long as it encounters line breaks in the current version of `buffer`. Pretty neat, huh?

### Handling text fields with embedded double quotes

Another challenge we are likely to encounter when reading 'messy' csv files is the inclusion of text fields (cells with longer strings of text in them) that might also have embedded double quotes, for example when the text includes a fragment where someone is quoted. More importantly, they might have embedded commas that separate different parts of a sentence. The problem with this is that we also use commas to distinguish between different columns of our csv file, and we somehow need to keep the two different types of comma separate. Let us first slightly adapt our example file to demonstrate what happens to the csv file if we embed double quotes and commas in a text field. 

*Just as a note on the side:* Depending on what text editor you use, you might not actually see the additional double quotes shown below. The csv standard prescribes that fields with embedded quotes are themselves quoted, and that single embedded quotes are always represented by two double quotes. However, some text editors may parse these double quotes out when you open the file, based on this standard.

``` text
timing,description,raw,comments,source
1,test1,"Let’s make this example",,"s1"
2,test2,"If there is one thing I hate, it is a single "" What about you?",,"s2"
3,test3,"""Nothing interesting really happened"", said Toby the magic purple elephant.",,"s3"
4,test4,"What about commas, huh?",,"s4"
```
In the second line below the header, you'll find a text cell with a single embedded double quote. As you can see, it is represented by two double quotes rather than one. In the second to last line of data, we find an example of someone being quoted. In the last line we find an embedded comma. For a simple csv parser, these are all quite challenging cases to deal with appropriately, especially if we find them in some combination. 

In the csv parser that I wrote, these cases are all handled by the `splitCsvLine(&tokens, buffer)` function. In the first block of code I included above, the function is called right after we have checked the lines for any line breaks. In other words, the version of buffer that is fed into the `splitCsvLine(&tokens, buffer)` function is already a complete version, where all line breaks have been dealt with. 

{% highlight c++ %}
/* 
   Function to split csv lines.
   Here, we should be careful with embedded quotes in text fields.
   Text fields in csv-files are marked by quotes.
   We repeatedly check whether or not we are in a text field and 
   respond accordingly.

   We also clean up the line of text a bit before we cut it up.
*/
void MainWindow::splitCsvLine(std::vector<std::string> *tokens,
			      std::string line) {
  bool inTextField = false; // To check whether or not we are in a text field.
  /* 
     The next two variables are to make sure that 
     we cut out clean segments of the line.
  */
  std::string::size_type stringLength = 0;
  std::string::size_type previousPos = 0;
  // We use this integer to count the number of quotes in a line so far.
  int quoteCount = 0; 
  for (std::string::size_type i = 0; i != line.length(); i++) {
    if (line[i] == '"') {
      quoteCount++; // If we found a quote, we increment this variable.
    }
    if (inTextField == false && line[i] == '"') {
      /* 
	 Finding a quote will mean that we are now 
	 in a text field if we weren't already.
      */
      inTextField = true;
      previousPos++; // We don't want to include the quote itself.
      stringLength--;
      /* 
	 If we are in a text field, and we encounter another quote, then we
	 should first check if it is accompanied by yet another pair of quotes. 
	 This is based on the fact that csv files (if written correctly), 
	 always use double quotes for
	 quotes that are embedded in text fields. 

	 We also double check by checking if the current amount of 
	 quotes is divisible by 2, that is, if we have an 
	 equal number of quotes.
      */
    } else if (inTextField == true && line[i] == '"' &&
	       line[i + 1] != '"' && quoteCount % 2 == 0) {
      // If these conditions hold, it means that we have now left the text field.
      inTextField = false;
      stringLength--; // We don't want to include the quote itself.
    }
    // If we encounter a comma, it means that we have encountered a new token.
    if (inTextField == false && line[i] == ',') {
      // We remove any white spaces before the start of this new token.
      while (line[previousPos] == ' ') {
	previousPos++;
	stringLength--;
      }
      // We create a substring here.
      std::string tempString = line.substr(previousPos, stringLength);
      // In case we have any remaining double quotes, let us remove them.
      if (tempString.size() > 1) {
	for (std::string::iterator it = tempString.begin() + 1; it != tempString.end();) {
	  if (*(it - 1) == '"' && *it == '"') {
	    tempString.erase(it);
	  } else {
	    it++;
	  }
	}
      }
      // And then we store the resulting string as a new token.
      tokens->push_back(tempString);
      previousPos = i + 1; // We set a new starting position for the next token.
      stringLength = 0; // And we reset the string length.
    } else {
      stringLength++; // We increment the string length as we go on.
    }
  }
  // After we finish reading the line, we should still include the last token.
  while (line[previousPos] == ' ') {
    previousPos++;
    stringLength--;
  }
  std::string tempString = line.substr(previousPos, stringLength);
  if (tempString.size() > 1) {
    for (std::string::iterator it = tempString.begin() + 1; it != tempString.end();) {
      if (*(it - 1) == '"' && *it == '"') {
	tempString.erase(it);
      } else {
	it++;
      }
    }
  }
  tokens->push_back(tempString);
}
{% endhighlight %}

This function reads through the object `buffer` (which is a string) character by character, and eventually pulls different substrings from the `buffer` object that we use as tokens to be written to our database. Near the beginning of this function, we define a bool variable `bool inTextField` that we use to check whether we are currently reading characters that are part of a text field. The idea behind this is very simple: If the bool is set to false, and we encounter a double quote, then this is taken as a sign that we are, from now on, inside a text field. As long as we are inside a text field, any commas we encounter will be ignored, and any double quotes we encounter will be handled in a special way. 

So, what happens once we read the last version of our example csv file, focusing only on the way that double quotes and embedded commas are handled? Let's start first with the line of data where we have one embedded quote. We first hit the double quote before the word "If", setting `quoteCount` to 1, and setting `inTextField` to true. We then hit the first double quote after the word "single". We set `quoteCount` to 2, and given that we are now in a text field, we check whether both the current character at position *i*, and the next character at position *i + 1* are double quotes. The `} else if () {` statement returns false, because we find a double quote at both positions, so we continue. We immediately encounter the second double quote after the word "single". Thus, we set `quoteCount` to 3, and check the conditional again. In this case, we do not find a double quote at position *i + 1* (we find a "." instead), but the last part of the conditional still makes it return false, since 3 % 2 does not equal 0. Just in case you don't know the meaning of the % symbol in this context, it means that we take the remainder after dividing `quoteCount` by 2. Whenever `quoteCount` is an even number, this statement will return 0. Given that the `} else if() {` still returns false, we continue again. We encounter the last quote of this line after "them.". We set `quoteCount` to 4, and we check the conditional again, which in this case returns true, since (1) we are currently in a text field, (2), at position *i* we find a double quote, and at position *i + 1* we do not, and the `quoteCount % 2` equals 0. This means that we have now (correctly) decided that we are no longer in a text field, and that we can continue treating commas and double quotes as we normally would. 

Let's consider the second to last line of data next. After two columns, we hit the three consecutive double quotes. The first of these sets the `inTextField` bool to true. The two other double quotes don't do anything but increment the `quoteCount` object, since the first of them is followed by another double quote, and since the second of them brings `quoteCount` to 3, an unequal number. We continue to the two double quotes after the word "happened", which again do nothing, since the first is followed by another double quote, and since the second brings `quoteCount` to an unequal number (5). The last double quote is encountered after "elephant.". This one brings `quoteCount` to a total of 6 for this line, and since it is not followed by another double quote, the function decides that we have finished reading the current text field. 

Now, let's discuss the last line of data in the file. This case is quite simple. Once we encounter the first double quote, `inTextField` is set to true, and therefore the comma we encounter after the word "commas" is ignored. After we encounter the second double quote, we set `inTextField` to false again, and the commas that we encounter in the remainder of the line are handled as delimiting commas. 

Thus, the problematic cases we discussed earlier are handled quite well. In fact, I am still quite surprised myself how flexible this set of functions is with messy data.

## Closing comments

As mentioned before, the functions discussed above take inspiration from various comments I encountered while browsing sites like Stack Overflow, and it would be nice to acknowledge everyone who has shared the knowledge that I made use of to write these functions, but I honestly do not remember where I found the different snippets of information. I guess I'll just have to thank the Stack Overflow community at large. 

The functions were written and repeatedly revised over quite a long period of time. After all this time, I had to study them in detail myself to understand their underlying logic again. There might still be important details that I forgot to mention here. However, in these cases "the proof of the pudding is in the eating." It should be easy to integrate these functions in your own software, and try them out for yourself. 

I hope they turn out to be useful to others!

[1]: http://www.indiana.edu/~socpsy/ESA/
[2]: {{ site.baseurl }}{% post_url 2017-03-10-bi-dynamic-line-graphs %}
[3]: https://pubsonline.informs.org/doi/abs/10.1287/orsc.1.3.313
[4]: https://en.wikipedia.org/wiki/Comma-separated_values
[5]: http://www.creativyst.com/Doc/Articles/CSV/CSV01.htm#FileFormat
