
  /* ---------------------- Variables --------------------------- */
  // Initialization ends on line: 37..

  let isGalleryDisplayed = true;
  let dropDownActive = false;

  let url_type = "";
  let modal_url = "";

  let project_filters = ["dl", "cert", "blog"];

  let social_navbar_text = [
    "#linkedin-navbar-text", 
    "#github-navbar-text", 
    "#medium-navbar-text", 
    "#publication-navbar-text", 
    "#twitter-navbar-text", 
    "#kaggle-navbar-text"
  ];

  let navbar_social_links = [
    "#linkedin-navbar-social-link",
    "#github-navbar-social-link",
    "#medium-navbar-social-link",
    "#publication-navbar-social-link",
    "#twitter-navbar-social-link",
    "#kaggle-navbar-social-link"
  ];

  // Project Titles
  let url2title = {
    "https://github.com/jlilleberg/Forecasting-Platinum-Palladium-Prices": "Forcasting Platinum and Palladium Prices",
    "https://github.com/jlilleberg/Malaria-Cell-Images-Classification": "Detecting Malaria Infected Bloodcells with Neural Networks",
    "https://github.com/jlilleberg/presidential-transcripts-analysis": "NLP Analysis of Presidential Transcripts",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization": "Deep Learning Specialization",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/NLP%20Specialization": "Natural Language Processing Specialization",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Developer%20Certificate": "TensorFlow Developer Certificate Specialization",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Data%20and%20Deployment%20Specialization": "TensorFlow Data and Deployment Specialization",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Data%20and%20Deployment%20Specialization/Course%201%20-%20Browser-based%20Models%20with%20TensorFlow.js": "TensorFlow Data and Deployment Specialization: Course 1",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Data%20and%20Deployment%20Specialization/Course%202%20-%20Device-based%20Models%20with%20TF%20Lite": "TensorFlow Data and Deployment Specialization: Course 2",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Data%20and%20Deployment%20Specialization/Course%203%20-%20Data%20Pipelines%20with%20TF%20Data%20Services": "TensorFlow Data and Deployment Specialization: Course 3",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Data%20and%20Deployment%20Specialization/Course%204%20-%20Advanced%20Deployment%20Scenarios%20with%20TF": "TensorFlow Data and Deployment Specialization: Course 4",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Developer%20Certificate/Course%201%20-%20Intro%20to%20TensorFlow": "TensorFlow Developer Specialization: Course 1",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Developer%20Certificate/Course%202%20-%20CNNs%20in%20TensorFlow": "TensorFlow Developer Specialization: Course 2",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Developer%20Certificate/Course%203%20-%20NLP%20in%20TensorFlow": "TensorFlow Developer Specialization: Course 3",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Developer%20Certificate/Course%204%20-%20Sequences%2C%20Time%20Series%2C%20and%20Prediction": "TensorFlow Developer Specialization: Course 4",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization/Course%201%20-%20Neural%20Networks%20and%20Deep%20Learning": "Deep Learning Specialization: Course 1",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization/Course%202%20-%20Improving%20Deep%20Neural%20Networks%20-%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization": "Deep Learning Specialization: Course 2",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization/Course%203%20-%20Structuring%20Machine%20Learning%20Projects": "Deep Learning Specialization: Course 3",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization/Course%204%20-%20Convolutional%20Neural%20Networks": "Deep Learning Specialization: Course 4",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization/Course%205%20-%20Sequence%20Models": "Deep Learning Specialization: Course 5",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/NLP%20Specialization/Course%201%20-%20NLP%20with%20Classification%20and%20Vector%20Spaces": "Natural Language Processing Specialization: Course 1",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/NLP%20Specialization/Course%202%20-%20NLP%20Processing%20with%20Probabilistic%20Models": "Natural Language Processing Specialization: Course 2",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/NLP%20Specialization/Course%203%20-%20NLP%20with%20Sequence%20Models": "Natural Language Processing Specialization: Course 3",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/NLP%20Specialization/Course%204%20-%20NLP%20with%20Attention%20Models": "Natural Language Processing Specialization: Course 4"
  };

  // Project Descriptions
  let url2description = {
    "https://github.com/jlilleberg/Forecasting-Platinum-Palladium-Prices": "Having wanted to learn time-series, I took an online class, read fpp2's online forecasting book and reviewed Facebook's Prophet. I pulled the current prices of Platinum from Quandl. Since this project was to improve my ability to forecast as well as the forecasting itself, I applied as many statistical concepts as possible to reinforce the strength of my forecasts and predictions.",
    "https://github.com/jlilleberg/Malaria-Cell-Images-Classification": "Malaria is a mosquito-borne infectious disease that affects humans and other animals. The symptoms range from tiredness, vomitting, and headacches to siezures, comas, and even death. Like any disease, being able to detect if a patient is infected is desireable. The dataset consists of 150 P. falciparum-infected and 50 healthy patients collected and photographed at Chittagong Medical College Hospital, Bangladesh.",
    "https://github.com/jlilleberg/presidential-transcripts-analysis": "The motivation for this project was to analyze presidential speeches throughout American history. In this end-to-end project, I scrapped and cleaned 992 transcripts were cleaned consisting of 3.8+ million words, or 22+ million characters. I then performed multiple analyses including sentiment analysis, text generation using deep neural networks, and a multitude of visualizations.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization": "In this specialization, I learned the foundations of Deep Learning, to understand how to build neural networks, and how to lead successful machine learning projects. I learned about Convolutional networks, RNNs, LSTM, Adam, Dropout, BatchNorm, Xavier/He initialization, and more. I worked on case studies from healthcare, autonomous driving, sign language reading, music generation, and NLP. I mastered not only the theory, but also see how it is applied in industry through Python and TensorFlow.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/NLP%20Specialization": "This specialization equipped me with the necessary skills in order to be ready to design NLP applications that perform question-answering and sentiment analysis, create tools to translate languages and summarize text, and even build chatbots. These and other NLP applications are going to be at the forefront of the coming transformation to an AI-powered future.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Developer%20Certificate": "This specialization taught me the necessary tools to build scalable AI-powered applications with TensorFlow. With these tools, I am now able to apply new TensorFlow skills to a wide range of problems and projects. This specialization also serves to help me prepare for the Google TensorFlow Certificate exam and thus, bringing me one step closer to achieving the Google TensorFlow Certificate.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Data%20and%20Deployment%20Specialization": "In this specialization, I learned how to get my machine learning models into the hands of real people on all kinds of devices by understanding how to train and run machine learning models in browsers and in mobile applications. I learned how to leverage built-in datasets with just a few lines of code, learned about data pipelines with TensorFlow data services, to use APIs to control data splitting, process all types of unstructured data and retrain deployed models with user data while maintaining data privacy. Lastly, I learned various deployment scenarios, TensorFlow Serving, TensorFlow, Hub, TensorBoard, and more.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Data%20and%20Deployment%20Specialization/Course%201%20-%20Browser-based%20Models%20with%20TensorFlow.js": "In this course, I learned to train and run machine learning models in any browser using TensorFlow.js, techniques for handling data in the browser, and how to build a computer vision project that recognizes and classifies objects from a webcam.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Data%20and%20Deployment%20Specialization/Course%202%20-%20Device-based%20Models%20with%20TF%20Lite": "In this course, I learned how to run my machine learning models in mobile applications and how to prepare models for a lower-powered, battery-operated devices, then execute models on both Android and iOS platforms. I also explored how to deploy on embedded systems using TensorFlow on Raspberry Pi and microcontrollers.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Data%20and%20Deployment%20Specialization/Course%203%20-%20Data%20Pipelines%20with%20TF%20Data%20Services": "In this course, I learned how use a suite of tools in TensorFlow to more effectively leverage data and train my model. I also learned how to leverage built-in datasets with just a few lines of code, use APIs to control how I split my data, and process all types of unstructured data.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Data%20and%20Deployment%20Specialization/Course%204%20-%20Advanced%20Deployment%20Scenarios%20with%20TF": "I explored four different scenarios I'll encounter when deploying models. I was introduced to TensorFlow Serving, a technology that lets me do inference over the web. Then TensorFlow Hub, a repository of models that me can use for transfer learning. Then I learned how to use TensorBoard to evaluate and understand how my models work, as well as share my model metadata with others. I also explored federated learning and how I can retrain deployed models with user data while maintaining data privacy.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Developer%20Certificate/Course%201%20-%20Intro%20to%20TensorFlow": "I learned how to use TensorFlow to implement the important and foundational principles of Machine Learning and Deep Learning so that I could start building and applying scalable models to real-world problems.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Developer%20Certificate/Course%202%20-%20CNNs%20in%20TensorFlow": "I learned advanced techniques to improve the computer vision model I built in Course 1. I explored how to work with real-world images in different shapes and sizes, visualize the journey of an image through convolutions to understand how a computer “sees” information, plot loss and accuracy, and explore strategies to prevent overfitting, including augmentation and dropout. Finally, I learned transfer learning and how learned features can be extracted from models.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Developer%20Certificate/Course%203%20-%20NLP%20in%20TensorFlow": "In this course, I learned to build natural language processing systems using TensorFlow. I learned to process text, including tokenizing and representing sentences as vectors, so that they can be input to a neural network. I also learned to apply RNNs, GRUs, and LSTMs in TensorFlow. Finally, I trained an LSTM on existing text to create original poetry.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Developer%20Certificate/Course%204%20-%20Sequences%2C%20Time%20Series%2C%20and%20Prediction": "In this course, I learned how to build time series models in TensorFlow. I first learned how to implement best practices to prepare time series data. I also explore how RNNs and 1D ConvNets can be used for prediction. Finally, I applied everything I learned throughout the Specialization to build a sunspot prediction model using real-world data.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization/Course%201%20-%20Neural%20Networks%20and%20Deep%20Learning": "I learned the major technology trends driving Deep Learning, built, trained, and applied fully connected deep neural networks, implemented efficient (vectorized) neural networks, what the key parameters in a neural network's architecture are.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization/Course%202%20-%20Improving%20Deep%20Neural%20Networks%20-%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization": "I learned the \"magic\" of getting deep learning to work well. Rather than the deep learning process being a black box, I spent time understanding what drives performance, and how to be able to more systematically get good results.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization/Course%203%20-%20Structuring%20Machine%20Learning%20Projects": "I learned how to build a successful machine learning project.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization/Course%204%20-%20Convolutional%20Neural%20Networks": "I learned how to build convolutional neural networks and apply it to image data.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization/Course%205%20-%20Sequence%20Models": "I learned how to build models for natural language, audio, and other sequence data.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/NLP%20Specialization/Course%201%20-%20NLP%20with%20Classification%20and%20Vector%20Spaces": "In this course, I learned to perform sentiment analysis of tweets using logistic regression and then naïve Bayes, use vector space models to discover relationships between words and use PCA to reduce the dimensionality of the vector space and visualize those relationships, and write a simple English to French translation algorithm using pre-computed word embeddings and locality sensitive hashing to relate words via approximate k-nearest neighbor search.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/NLP%20Specialization/Course%202%20-%20NLP%20Processing%20with%20Probabilistic%20Models": "In this course, I learned to create a simple auto-correct algorithm using minimum edit distance and dynamic programming, apply the Viterbi Algorithm for part-of-speech (POS) tagging, which is important for computational linguistics, write a better auto-complete algorithm using an N-gram language model, and write your own Word2Vec model that uses a neural network to compute word embeddings using a continuous bag-of-words model.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/NLP%20Specialization/Course%203%20-%20NLP%20with%20Sequence%20Models": "In this course, I learned to train a neural network with GLoVe word embeddings to perform sentiment analysis of tweets, generate synthetic Shakespeare text using a Gated Recurrent Unit (GRU) language model, train a recurrent neural network to perform named entity recognition (NER) using LSTMs with linear layers, and use so-called ‘Siamese’ LSTM models to compare questions in a corpus and identify those that are worded differently but have the same meaning.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/NLP%20Specialization/Course%204%20-%20NLP%20with%20Attention%20Models": "In this course, I learned to translate complete English sentences into German using an encoder-decoder attention model, build a Transformer model to summarize text, use T5 and BERT models to perform question-answering, and build a chatbot using a Reformer model."
  };

  /* ------------------- Filter Projects ------------------------ */
  

  // Displays all dl projects
  function displayDeepLearningProjects(){
    $("#gallery").fadeOut("slow");
    $("body").css("pointer-events", "none");
    setTimeout(function(){

      // Hide all items in gallery
      $.each($("#gallery-projects").children(), function(index, value){
        $(this).addClass("hide-project");
      });

      // Show all certifications
      $.each($("#gallery-projects").children(), function(index, value){
        if ($(this).hasClass("dl")) {
          $(this).removeClass("hide-project");
        }
      });

      // Fade In Gallery
      $("#gallery").fadeIn("slow");

      // Enable pointer
        setTimeout(function(){
          $("body").css("pointer-events", "auto");
        }, 100);
    }, 600);
  };

  // Displays all certifications
  function displayCertifications(){
    $("#gallery").fadeOut("slow");
    $("body").css("pointer-events", "none");
    setTimeout(function(){

      // Hide all items in gallery
      $.each($("#gallery-projects").children(), function(index, value){
        $(this).addClass("hide-project");
      });

      // Show all certifications
      $.each($("#gallery-projects").children(), function(index, value){
        if ($(this).hasClass("cert")) {
          $(this).removeClass("hide-project");
        }
      });

      // Fade In Gallery
      $("#gallery").fadeIn("slow");

      // Enable pointer
        setTimeout(function(){
          $("body").css("pointer-events", "auto");
        }, 100);
    }, 600);
  };

  // Displays all blog projects
  function displayBlogs(){
    $("#gallery").fadeOut("slow");
    $("body").css("pointer-events", "none");
    setTimeout(function(){

      // Hide all items in gallery
      $.each($("#gallery-projects").children(), function(index, value){
        $(this).addClass("hide-project");
      });

      // Show all certifications
      $.each($("#gallery-projects").children(), function(index, value){
        if ($(this).hasClass("blog")) {
          $(this).removeClass("hide-project");
        }
      });

      // Fade In Gallery
      $("#gallery").fadeIn("slow");

      // Enable pointer
        setTimeout(function(){
          $("body").css("pointer-events", "auto");
        }, 100);
    }, 600);
  };

  // Display all projects
  function displayAllProjects(){

     // Fade out gallery and disable pointer
    $("#gallery").fadeOut("slow");
    $("body").css("pointer-events", "none");
    setTimeout(function(){

      // Show all projects
      $(".all").removeClass("hide-project");

      // Fade in gallery
      $("#gallery").fadeIn("slow");
      
      // Enable pointer
      setTimeout(function(){
        $("body").css("pointer-events", "auto");
      }, 100);
    }, 600);
  };

  // Filter Handler
  $(function(){
    let selectedClass = "";
    $(".filter").click(function(){

      // Get respective filter
      selectedClass = $(this).attr("data-rel");

      // Fillter projects
      if (selectedClass == "all") {
        displayAllProjects();
      } else if (selectedClass == "dl") {
        displayDeepLearningProjects();
      } else if (selectedClass == 'cert') {
        displayCertifications();
      } else if (selectedClass == "blog") {
        displayBlogs();
      };
    });
  });

  // Toggle boolean for if mobile navbar is active
  $(function(){
    $("#navbar-mobile-btn").click(function(){
      dropDownActive = !dropDownActive;
      if (dropDownActive) {
        $("#gallery-container").addClass("hide-gallery");
      } else {
        $("#gallery-container").removeClass("hide-gallery");
      };
    });
  });

  // Modal is clicked, blur everything else
  $(function(){
    $(".enable-blur").click(function(){
      $(".blur-candidate").addClass("blur-element");
      $(this).removeClass("blur-element");
    });
  });

  // Remove blur when modal is hidden
  $('#ModalCenter').on('hide.bs.modal', function (e) {
    $(".blur-candidate").removeClass("blur-element");
  });

  // If portfolio is clicked and we are already on portfolio, close navbar dropdown
  $(function(){
    $('#portfolio-link').click(function(){
      $("#navbar-mobile-btn").click();
    });
  });


  /* ------------------- Modal Projects Images and Links -------------------*/

  // Get and set URL for selected project
  $(function(){
    $(".project-item").click(function(){

      // Get url type and url link
      url_type = $(this).attr("url_type");
      modal_url = $(this).attr("url");

      
      $("#modal-button-github-id").attr("href", modal_url);    

      // Update modal title and modal text
      $("#modal-title-id").text(url2title[modal_url]);
      $("#modal-text-id").text(url2description[modal_url]);

      // Update Modal image to respective project image
      image_src = $(this).find("a div img").attr("src");
      $("#modal-image-id").attr("src", image_src);
    });
  });

  // Github Button is Pressed
  $(function(){
    $("#modal-button-github-id").click(function(){
      let win = window.open(modal_url, "_blank");
      if (win) {
        win.focus();
      }
    })
  })

  /* ------------------- Document is Ready ---------------------- */

  $(document).ready(function () {

    /* ---------------------- DOM is Ready -------------------------- */

    // Large Devices: Add text to social media buttons and appropriate css styles
    if($(this).width() > 1550) {
      $("#linkedin-navbar-text").text("Linkedin ");
      $("#github-navbar-text").text("Github ");
      $("#medium-navbar-text").text("Medium ");
      $("#publication-navbar-text").text("Publication ");
      $("#twitter-navbar-text").text("Twitter ");
      $("#kaggle-navbar-text").text("Kaggle ");

      // Update class for social buttons with text
      $.each(navbar_social_links, function(index, value) {
        $(value).addClass("navbar-social-btn-width-with-text");
      });

    // Medium or smaller device:
    } else {

      // Hide additional modal text for smaller screens
      $(".proj-image-text-container").addClass("hide-obj");
      $(".proj-image-title").css("top", "35%");

      // Remove text for social buttons
      $.each(social_navbar_text, function(index, value) {
        $(value).text("");
      });

      // Apply appropriate css styles
      $.each(navbar_social_links, function(index, value) {
        $(value).addClass("navbar-social-btn-width-with-no-text");
      });
    }

    /* ---------------------- Resizing ---------------------------- */

    // When resizing, check width and adjust accordingly
    $(function(){
      $(window).on('resize', function(){

        // For large devices
        if($(this).width() > 1550) {

          // Add labels to social media buttons
          $("#linkedin-navbar-text").text("Linkedin ");
          $("#github-navbar-text").text("Github ");
          $("#medium-navbar-text").text("Medium ");
          $("#publication-navbar-text").text("Publication ");
          $("#twitter-navbar-text").text("Twitter ");
          $("#kaggle-navbar-text").text("Kaggle ");

          // Apply css changes for social media buttons with text
          $.each(navbar_social_links, function(index, value){

            if ($(value).hasClass("navbar-social-btn-width-with-no-text")) {
              $(value).removeClass("navbar-social-btn-width-with-no-text");
            };

            $(value).addClass("navbar-social-btn-width-with-text");
          });

          // Display additional modal iamge text for larger screens
          $(".proj-image-text-container").removeClass("hide-obj");
          $(".proj-image-title").css("top", "20%");

        // For medium or smaller devices
        } else {

          // Hide additional modal image text for small screens
          $(".proj-image-text-container").addClass("hide-obj");
          $(".proj-image-title").css("top", "35%");

          // Remove text from social media buttons
          $.each(social_navbar_text, function(index, value) {
            $(value).text("");
          });

          // Apply css changes for social media buttons without text
          $.each(navbar_social_links, function(index, value){
            if ($(value).hasClass("navbar-social-btn-width-with-text")) {
              $(value).removeClass("navbar-social-btn-width-with-text");
            };

            $(value).addClass("navbar-social-btn-width-with-no-text");
          });
        };          
      });
    }); 

    //  If window is resized to a larger width with the navbar dropdown active, de-activate navbar dropdown
    $(function(){
      $(window).on('resize', function(){
        if($(this).width() > 991.98 && dropDownActive == true) {
          $("#navbar-mobile-btn").click();
        };
      });
    });


    $('.navbar-dropdown-toggle-button').on('click', function () {
      $('.navbar-dropdown-animation').toggleClass('open');
    });

    $(function () {
      $('[data-toggle="tooltip"]').tooltip();
    });
  });