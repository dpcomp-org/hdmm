import * as util from './util.js';

var dataFile = new Array()
var codeFile1 = new Array();
var codeFile2 = new Array();
var codeFile3 = new Array();
var expectedError1 = new Array();
var expectedError2 = new Array();
var expectedError3 = new Array();
var domainMeta = new Array();
var workloadsObj = new Array();
var workloadsBuilder = new Array();
var buildingBlocksNum = 0;
var workloadNum = 0;
var csvParser = new Array();
var pdfChart;
var cdfChart;
var wlpdfChart;
var barChart;
var hdmm3,identity = [];
var workloadsData = new Object;
var eps = 1.0;
const colorPattern = ['#ff2e63','#5e0606','#48466d', '#eac100' ,'#30e3ca','#fcd539', '#f07143']

$(document).ready(function() {

	// Constant div card 
	const DIV_CARD = 'div.card';
	// collapse card
	$('[data-toggle="card-collapse"]').on('click', function(e) {
		let $card = $(this).closest(DIV_CARD);

		$card.toggleClass('card-collapsed');

		e.preventDefault();
		return false;
	});

	// Domain Card JS //
	//If user choose to upload sample csv...
	$("#exampleBtn").click(function(){
		$("#initialCard").find('[data-toggle="card-collapse"]').click();
		$("#initialCard #cardStatus").removeClass().addClass('fa fa-check-circle');
		$("#domainCard").removeClass('card-collapsed').addClass('card-collapse');
		$('.bootstrap-tagsinput input').prop('readonly', true);
		$("#domainInput").empty();
		$.ajax({
			method: "GET",
			dataType: 'json',
			contentType: "application/json",
			async: true,
			url:"/api/upload",
			success:function(res){
				$("#domainCard").show();
				if (res.error == 0) {
					$("#domainUpload").show();
					$("#domainInput").empty().hide();
					$("#domainTable").empty();
					domainDataTable(res);
					csvParser = res
				};
			}
		});
	});

	$("#uploadBtn").click(function(){
		$("#fileSelection").click();
		$("#fileSelection").on('change',function(){
			if ($(this).val().split('\\').pop() != "") {
				var sampleCSV = $(this).prop('files')[0];
				var data = new FormData();
				$("#initialCard .alert").alert('close');
				data.append('file', sampleCSV);
				$.ajax({
					method: "POST",
					data: data,
					dataType : 'json',
				    cache: false,
				    contentType: false,
				    processData: false,
					url:"/api/upload",
					success:function(res){
						if (res.error == 0) {
							$("#initialCard").find('[data-toggle="card-collapse"]').click();
							$("#initialCard #cardStatus").removeClass().addClass('fa fa-check-circle');
							$("#domainCard").removeClass('card-collapsed').addClass('card-collapse');
							$('.bootstrap-tagsinput input').prop('readonly', true);
							$("#domainInput").empty();
							$("#domainCard").show();
							$("#domainUpload").show();
							$("#domainInput").empty().hide();
							$("#domainTable").empty();
							domainDataTable(res);
							csvParser = res
						} else {
							$("#initialCard #alertMessage").html(util.uploadFail.replace(/{{error-msg}}/g, res.msg));
						};
					}
				});
			};
		});
		return;
	});

	//If use choose to type in domain info...
	$("#inputBtn").click(function(){
		$("#initialCard").find('[data-toggle="card-collapse"]').click();
		$("#initialCard #cardStatus").removeClass().addClass('fa fa-check-circle');
		$("#domainCard").removeClass('card-collapsed').addClass('card-collapse');
		$('.bootstrap-tagsinput input').prop('readonly', true);
		$("#domainCard").show();
		$("#domainTable").empty();
		$("#domainUpload").hide();
		$("#domainInput").empty().append(util.domainInput).show();
		$('.selectpicker').selectpicker({hideDisabled: false});
	});

	//change type of domain
	$("#domainTable").on('change', "[id=domainType]", function(){
		var formId = $(this).closest('form').attr('id');
		var table = $('#domainTable').DataTable();
		var tr = $(this).closest("tr").prev()[0];
		var row = table.row( tr );
		if($(this).val() == 'Categorical'){
			if(row.data().type == "Categorical"){
				row.child( util.domainEditorCategorical
					.replace(/{{form-id}}/g, formId)
					.replace(/{{distinct-value}}/g,row.data().value)).show();
			} else {
				row.child( util.domainEditorCategorical
					.replace(/{{form-id}}/g, formId)
					.replace(/{{distinct-value}}/g,'')).show();
			};
		};
		if($(this).val() == 'Numerical'){
			row.child( util.domainEditorNumerical
				.replace(/{{form-id}}/g, formId)
				.replace(/{{lower-bound}}/g, row.data().min)
				.replace(/{{upper-bound}}/g, row.data().max)
				.replace(/{{bucket-size}}/g, Math.ceil((row.data().max - row.data().min)/100))).show()
		};
		$.fn.selectpicker.Constructor.BootstrapVersion = '4';
		$('.selectpicker').selectpicker({hideDisabled: false});
	});

	//add selected domin as tagsinput
	$("#domainTable").on('click', "[id=addDomain]", function(){
		var formObject = $(this).closest('form');
		var domainName = formObject.attr('id').replace("domainEditor-", "");
		var domainType = $("#domainEditor-" + domainName + " #domainType option:selected").attr('value');
		var inputCheck = appendDomainFromTable(domainName, domainType);
		if(inputCheck){
			if (domainMeta.length == 1){
				$("#domainCard #domainSubmit").html(util.domainSubmit)
				$("#selectedDomains").tagsinput({
		  			freeInput: false,
		  			allowDuplicates: false,
		  			onTagExists: function(item, $tag) {}
				});
			};
			
			$('#selectedDomains').tagsinput('add',domainName);
			var table = $('#domainTable').DataTable();
			var tr = $(this).closest("tr").prev()[0];
			var row = table.row( tr );
			row.child.hide();
			$(row.node()).removeClass('shown');
		};
	});

	$("#domainInputCategorical, #domainInputNumerical").on('click', '[id=addDomain]', function(){
		var domainType = $(this).attr('data-type');
		var domainName = $(this).attr('data-name');
		var inputCheck = appendDomainFromModal(domainName, domainType);
		if(inputCheck){
			if (domainMeta.length == 1){
				$("#domainCard #domainSubmit").html(util.domainSubmit)
				$("#selectedDomains").tagsinput({
		  			freeInput: false,
		  			allowDuplicates: false,
		  			onTagExists: function(item, $tag) {}
				});
			};
			$(this).closest('.modal').modal('toggle');
			$('#selectedDomains').tagsinput('add',domainName);
		};
	});

	//add remove domin from domainMeta array
	$(document).on('click', "span[data-role='remove']", function(){
		var badge = $(this).closest('.badge-info');
		removeDomain(badge.text());
		if (domainMeta.length == 0){
			$("#domainCard #domainSubmit").html('<p style="font-style:italic;text-align: center;"> 0 domain selected.</p>')
		};
	});

	$("#domainInput").on('click','[id=callDomainModal]',function(){
		var domainName = $("#domainInput #inputDomainName").val();
		var domainType = $("#domainInput #inputdomainType option:selected").val();
		if (domainName != "" && domainType == "Numerical"){
			$("#domainInputNumerical .modal-title").html("Add numerical attribute - " + domainName);
			$("#domainInputNumerical #addDomain").attr("data-name", domainName)
			$("#domainInputNumerical #addDomain").attr("data-type", domainType)
			$("#domainInputNumerical #lowerBound, #upperBound, #bucketSize").val('');
			$("#domainInputNumerical #lowerBound, #upperBound, #bucketSize").removeClass('highlight');
			$("#domainInputNumerical").modal();
		} else {
			$("#domainInputCategorical .modal-title").html("Add categorical attribute - " + domainName);
			$("#domainInputCategorical #addDomain").attr("data-name", domainName)
			$("#domainInputCategorical #addDomain").attr("data-type", domainType)
			$("#domainInputCategorical #valueList").val('');
			$("#domainInputCategorical #isOrdered").prop('checked', false);
			$("#domainInputCategorical").modal();
		};
		$("#domainInput #inputDomainName").val('');
	});

	// Workload Card JS//
	//event triggered by submitting domain: from domain card to workload card
	$("#workloadCard").on('click',"[id=typeGeneralWorkload]",function(){
		$("#chooseWorkload").hide();
		$("#generalWorkload").show();
		$("#marginalWorkload").hide();
		$("#backToWorkloadType").show();
		$("#buildingBlocks").empty();
		var options = domainOptions();
		var buildingBlockEditor = util.buildingBlockEditor.replace(/{{domain-options}}/g,options)
									.replace(/{{building-block-number}}/g,'I');
		$("#workloadCard #buildingBlocks").append(buildingBlockEditor);
		buildingBlocksNum = 1;
		workloadNum = 0;
		$('.selectpicker').selectpicker({hideDisabled: false});
		$("#workloadSubmit").html('<p style="font-style:italic; text-align: center;"> 0 workload created.</p>');
		$("#workloadSubmitButton").hide();
		workloadsObj = [];
	});

	$("#workloadCard").on('click',"[id=typeMarginalWorkload]",function(){
		$("#chooseWorkload").hide();
		$("#marginalWorkload").show();
		$("#generalWorkload").hide();
		$("#backToWorkloadType").show();
		$("#marginalBlocks").empty();
		var options = domainCheckbox();
		$("#workloadCard #marginalBlocks").append(options);
	});

	$("#workloadCard").on('click',"[id=backToWorkloadType]",function(){
		$("#chooseWorkload").show();
		$("#marginalWorkload").hide();
		$("#generalWorkload").hide();
		workloadsObj = [];
		buildingBlocksNum = 1;
		workloadNum = 0;
		$("#backToWorkloadType").hide();
		$("#workloadSubmit").empty().html('<p style="font-style:italic; text-align: center;"> 0 workload created.</p>');
		$("#workloadSubmitButton").hide();
		$("#workloadBodyWrap").collapse('show');
		$("#codeCard").hide();
		$("#exportCard").hide();
		$("#summaryCard").hide();
	});

	$("#domainSubmit").on('click',"[id=domainSubmitButton]",function(){
		$("#domainBodyWrap").collapse('hide');
		$("#domainCard #cardStatus").removeClass().addClass('fa fa-check-circle');
		$("#workloadCard").show();
		$("#workloadBodyWrap").collapse('show');
		$("#summaryCard").removeClass('card-collapsed').addClass('card-collapse');
		$("#summaryCard").hide();
		$("#marginalWorkload").hide();
		$("#generalWorkload").hide();
		$("#chooseWorkload").show();
		workloadsObj = [];
		buildingBlocksNum = 1;
		workloadNum = 0;
		$("#workloadSubmit").empty().html('<p style="font-style:italic; text-align: center;"> 0 workload created.</p>');
		$("#workloadSubmitButton").hide();
		$("#codeCard").hide();
		$("#exportCard").hide();
	});

	$("#domainMetaButton").on('click',function(){
		var domainConfigs = '';
		for (var index = 0; index < domainMeta.length; ++index) {
			domainConfigs += '<tr>';
			domainConfigs += '<td>' + domainMeta[index].name +'</td>';
			if(domainMeta[index].type == 'Numerical'){
				domainConfigs += '<td>' + JSON.stringify({'type':domainMeta[index].type, 'lowerBound':domainMeta[index].minimum,'upperBound':domainMeta[index].maximum,'bucketSize':domainMeta[index].bucketSize}, null, 2) +'</td>';
			} else {
				domainConfigs += '<td>' + JSON.stringify({'type':domainMeta[index].type, 'domainValues':domainMeta[index].values}, null, 2) +'</td>';
			};
			domainConfigs += '</tr>';
		};
		$("#domainMetaModal #domainMetaTable").html(util.domainMetaTable.replace(/{{domain-configs}}/g,domainConfigs));
		$("#domainMetaModal").modal();
	});

	//add building block
	$("#addBuildingBlock").on('click', function(){
		buildingBlocksNum += 1;
		var options = domainOptions();
		var roman = romanize(buildingBlocksNum);
		var buildingBlockEditor = util.buildingBlockEditor.replace(/{{domain-options}}/g,options)
									.replace(/{{building-block-number}}/g, roman);
		if (buildingBlocksNum != 1){
			$("#workloadCard #buildingBlocks").append(util.kroneckerDiv)
		};
		$("#workloadCard #buildingBlocks").append(buildingBlockEditor);
		$('.selectpicker').selectpicker({hideDisabled: false});
		if (buildingBlocksNum > 1){
			$('#removeBuildingBlock').show();
		};
		distinctDomain();
	});

	//remove building block
	$("#removeBuildingBlock").on('click', function(){
		if (buildingBlocksNum != 1){
			var roman = romanize(buildingBlocksNum);
			var buildingBlockId = "#buildingBlock" + roman;
			$("#workloadCard").find('.kroneckerdiv:last').remove();
			$(buildingBlockId).remove();
			buildingBlocksNum -= 1;
		};
		if (buildingBlocksNum == 1) {
			$('#removeBuildingBlock').hide();
		};
		distinctDomain();
	});

	//generate building block strings
	$("#workloadCard").on('change', '[id="blockType"], [id="blockDomain"]', function(){
		var buildingBlockId = $(this).closest('form').attr('id');
		var blockType = $('#' + buildingBlockId + ' #blockType option:selected').val();
		var domain = $('#' + buildingBlockId + ' #blockDomain option:selected').val();
		var index = domainMeta.findIndex(function(item, i){
	  		return item.name == domain
		});
		if (blockType != "" && domain != "" && blockType != "customize"){
			var buildingBlockString = blockString(blockType,domain)
			$('#' + buildingBlockId + ' #blockSummary').val(buildingBlockString);
			var temp = $.extend(true,{},domainMeta[index]);
			temp.buildingBlockId = buildingBlockId;
			temp.buildingBlock = blockType;
			temp.buildingBlockString = buildingBlockString;
			index = indexFinder(workloadsBuilder, 'buildingBlockId', buildingBlockId);
			if(index == -1){
				workloadsBuilder.push(temp);
			} else {
				workloadsBuilder[index] = temp;
			};
		};
		if (domain != "" && blockType == "customize"){
			$('#' + buildingBlockId + ' #blockSummary').val('');
			$("#domainCustomize .modal-title").html('Customize buildingblock for domain - ' + domain);
			$("#domainCustomize #addCustomizedBB").attr('data-name', domain);
			$("#domainCustomize #addCustomizedBB").attr('data-id', buildingBlockId);
			if(domainMeta[index].type == 'Categorical'){
				$("#domainCustomize #valueList").val(domainMeta[index].values);
			} else {
				var arr = range(parseInt(domainMeta[index].minimum),
								parseInt(domainMeta[index].maximum),
								parseInt(domainMeta[index].bucketSize));
				$("#domainCustomize #valueList").val(arr.join());
			};
			$("#domainCustomize input[name=customizeQueries]").val('');
			$("#domainCustomize input[name=customizeName]").val('');
			$("#domainCustomize #alertMessage").empty();
			$('#domainCustomize .entry').not(':first').remove();
			$('#domainCustomize .entry .btn').removeClass('btn-remove').addClass('btn-add')
											 .removeClass('btn-danger').addClass('btn-success')
											 .html('<i class="fas fa-plus"></i>');
			$("#domainCustomize").modal();
		};
		distinctDomain();
	});

	$("#domainCustomize").on('click', '.btn-add', function(){
		var controlForm = $(this).closest('.form-group'),
		currentEntry = $(this).parents('.entry:first'),
		newEntry = $(currentEntry.clone().css('padding-top', '20px')).appendTo(controlForm);
		newEntry.find('input').val('');
		controlForm.find('.entry:not(:last) .btn-add')
			.removeClass('btn-add').addClass('btn-remove')
			.removeClass('btn-success').addClass('btn-danger')
			.html('<i class="fas fa-minus"></i>');
	}).on('click', '.btn-remove', function(e){
		$(this).parents('.entry:first').remove();
		return false;
	});

	$("#domainCustomize #addCustomizedBB").on('click', function(){
		var domain = $(this).attr('data-name');
		var index = indexFinder(domainMeta, 'name', domain);
		var buildingBlockId = $(this).attr('data-id');
		var customized = [],parsed = []; 
		$("input[name=customizeQueries]").each(function (index, element ) {
			if ($(this).val() != ""){
				parsed = customizedParser($(this).val().split(","),$("#domainCustomize #valueList").val().split(","),domain);
				if(!parsed){
					customized = false;
					return
				} else {
					customized.push(parsed);
				};
			};
		});
		var buildingBlockName = $("input[name=customizeName]").val();
		if (!customized){
			$("#domainCustomize #alertMessage").html(util.customizeInputError.replace(/{{error-msg}}/g,"Value error, please check your input."));
		} else if (buildingBlockName == ""){
			$("#domainCustomize #alertMessage").html(util.customizeInputError.replace(/{{error-msg}}/g,"Please name your buildingblock."));
		} else {
			var temp = $.extend(true,{},domainMeta[index]);
			temp.buildingBlockId = buildingBlockId;
			temp.buildingBlock = 'customized';
			var buildingBlockString = blockString('customized', domain, buildingBlockName)
			$('#' + buildingBlockId + ' #blockSummary').val(buildingBlockString);
			temp.buildingBlockName = buildingBlockName;
			temp.customizedQueries = customized;
			temp.buildingBlockString = buildingBlockString;
			index = indexFinder(workloadsBuilder, 'buildingBlockId', buildingBlockId);
			if(index == -1){
				workloadsBuilder.push(temp);
			} else {
				workloadsBuilder[index] = temp;
			};
			$(this).closest('.modal').modal('toggle');
		};
	});

	$("#workloadCard").on('change', '[id="blockDomain"]', function(){
		var buildingBlockId = $(this).closest('form').attr('id');
		var domain = $('#' + buildingBlockId + ' #blockDomain option:selected').val();
		if (domain != ""){
			var index = domainMeta.findIndex(function(item, i){
	  			return item.name == domain
			});
			if (domainMeta[index].type == 'Categorical'){
				$('#' + buildingBlockId + ' #blockType' + ' option[value=prefix]').prop('disabled', 'disabled');
				$('#' + buildingBlockId + ' #blockType' + ' option[value=allrange]').prop('disabled', 'disabled');
			} else {
				$('#' + buildingBlockId + ' #blockType' + ' option[value=prefix]').prop('disabled', false);
				$('#' + buildingBlockId + ' #blockType' + ' option[value=allrange]').prop('disabled', false);
			};
			$('#' + buildingBlockId + ' #blockType').prop('disabled', false);
			$('#' + buildingBlockId + ' #blockType').selectpicker('refresh');
			$('.selectpicker').selectpicker({hideDisabled: false});
		};
	});

	//add general workload
	$("#addWorkload").on('click', function(){
		for (var index = 0; index < domainMeta.length; ++index) {
			var indexfin = indexFinder(workloadsBuilder, 'name', domainMeta[index].name);
			if (indexfin == -1){
				var temp = $.extend(true,{},domainMeta[index]);
				temp.buildingBlockString = blockString(temp.buildingBlock, temp.name);
				workloadsBuilder.splice(index, 0, temp);
			} else if (indexfin > index){
				[workloadsBuilder[indexfin], workloadsBuilder[index]] = [workloadsBuilder[index], workloadsBuilder[indexfin]];
			};
		};
		temp = []
		for (var index = 0; index < domainMeta.length; ++index) {
			domainMeta[index].name
		};

		var stringObj = new Array();
		for (index = 0; index < workloadsBuilder.length; ++index) {
			stringObj.push(workloadsBuilder[index].buildingBlockString);
		};
		workloadNum += 1;
		var workloadString = stringObj.join(' &#8855; ');
		workloadsObj.push(workloadsBuilder);

		$("#buildingBlocks").empty();
		$('#removeBuildingBlock').hide();
		buildingBlocksNum = 1;
		var options = domainOptions();
		var buildingBlockEditor = util.buildingBlockEditor.replace(/{{domain-options}}/g,options)
									.replace(/{{building-block-number}}/g,'I');
		$("#workloadCard #buildingBlocks").append(buildingBlockEditor);
		$('.selectpicker').selectpicker({hideDisabled: false});
		var workloadRow = util.workloadRow.replace(/{{workloadRow-number}}/g, romanize(workloadNum))
						.replace(/{{workloadRow-string}}/g, workloadString)
						.replace(/{{workloadRow-weight}}/g, 1.0.toFixed(1));
		if(workloadNum == 1){
			$("#workloadSubmit").empty();
			$("#workloadSubmit").append(workloadRow);
			$("#workloadSubmitButton").show();
		}
		else {
			$("#workloadSubmit").append(workloadRow);
		};
		workloadsBuilder = [];
		$("#workloadSubmit div[id^=workloadRow]").each(function () {
			$(this).find("#workloadWeight").val(1.0.toFixed(1));
		});
		//check workload total size
		var oversize = sizeCalculator();
		if (oversize){
			$("#submitWorkload").prop("disabled",true);
			$(this).prop("disabled",true);
			$("#workloadCard .card-body").prepend('<div class="alert alert-danger" role="alert">Workloads oversize, please try smaller workloads.</div>');
		};
	});

	//add marginal workload
	$("#addMarginls").on('click', function(){
		for (var index = 0; index < domainMeta.length; ++index) {
			var temp = $.extend(true,{},domainMeta[index]);
			if ($("#domainCheckbox-" + domainMeta[index].name).is(':checked')){
				temp.buildingBlockString = blockString('identity', temp.name);
				temp.buildingBlock = 'identity';
				workloadsBuilder.splice(index, 0, temp);
				$("#domainCheckbox-" + domainMeta[index].name).prop('checked', false);
			} else {
				temp.buildingBlockString = blockString('total', temp.name);
				workloadsBuilder.splice(index, 0, temp);
			}
		};
		temp = []
		for (var index = 0; index < domainMeta.length; ++index) {
			domainMeta[index].name
		};
		
		var stringObj = new Array();
		for (index = 0; index < workloadsBuilder.length; ++index) {
			stringObj.push(workloadsBuilder[index].buildingBlockString);
		};
		workloadNum += 1;
		var workloadString = stringObj.join(' &#8855; ');
		workloadsObj.push(workloadsBuilder);

		var workloadRow = util.workloadRow.replace(/{{workloadRow-number}}/g, romanize(workloadNum))
						.replace(/{{workloadRow-string}}/g, workloadString)
						.replace(/{{workloadRow-weight}}/g, 1.0.toFixed(1));
		if(workloadNum == 1){
			$("#workloadSubmit").empty().append(workloadRow);
			$("#workloadSubmitButton").show();
		}
		else {
			$("#workloadSubmit").append(workloadRow);
		};
		$("#workloadSubmit div[id^=workloadRow]").each(function () {
			$(this).find("#workloadWeight").val(1.0.toFixed(1));
		});

		workloadsBuilder = [];
		//check workload total size
		var oversize = sizeCalculator();
		if (oversize){
			$("#submitWorkload").prop("disabled",true);
			$(this).prop("disabled",true);
			$("#workloadCard .card-body").prepend('<div class="alert alert-danger" role="alert">Workloads oversize, please try smaller workloads.</div>');
		};
	});

	//delete workload
	$("#workloadSubmit").on('click', '[id="removeWorkload"]', function(){
		var indexClicked = $("#workloadSubmit #removeWorkload").index(this);
		workloadsObj.splice(indexClicked, 1);
		var workloadRowId = $(this).closest('div[id^="workloadRow"]').attr('id');
		$('#' + workloadRowId).remove();
		workloadNum -= 1;
		$("#workloadSubmit div[id^=workloadRow]").each(function (index, element ) {
			$(this).find("#workloadWeight").val(1.0.toFixed(1));
			$(this).attr("id","workloadRow" + romanize(index+1));
			$(this).find('label[for="selectedWorkload"]').text('Workload ' + romanize(index+1));
		});
		if (workloadNum == 0) {
			$("#workloadSubmit").empty().append('<p style="font-style:italic; text-align: center;"> 0 workload created.</p>');
			$("#workloadSubmitButton").hide();
		};
		var oversize = sizeCalculator();
		if (!oversize){
			$("#submitWorkload").prop('disabled', false);
			$("#addWorkload").prop('disabled', false);
			$("#addMarginls").prop('disabled', false);
			$("#workloadCard .card-body .alert").remove();
		};
	});

	//allocate weights
	$("#workloadSubmit").on('change', '[id="workloadWeight"]', function(){
		//TODO
	});

	//submit worloads
	$("#submitWorkload").on('click',function(){
		var workloads = new Array();
		hdmm3 = [];
		identity = [];
		$('#hdmmProgress').show();
		$('#hdmmProgress .progress-bar').css('width', '100%').attr('aria-valuenow', '100').html('Starting');
		$('#hdmmTable').hide();
		$('#summaryTable').hide();
		$('#pdfChart').hide();
		$('#cdfChart').hide();
		$('#wlpdfChart').hide();
		$("#codeCard").hide();
		$("#exportCard").hide();
		$('#epsilon').val(1.0.toFixed(1));
		$("#workloadSubmit div[id^=workloadRow]").each(function (index, element ) {
			var workloadRowId = $(this).attr('id');
			var workloadString = $('#' + workloadRowId + ' #selectedWorkload').val();
			var workloadWeight = $('#' + workloadRowId + ' #workloadWeight').val();
			workloads.push({'meta':domainMeta, 'filename':csvParser.filename, 'wid':workloadRowId,'data':workloadsObj[index],'weight':workloadWeight,'workloadString':workloadString.replace(/ ⊗ /g,'+')});
		});
        expectedError1 = Number.MAX_VALUE;
        expectedError2 = Number.MAX_VALUE;
        expectedError3 = Number.MAX_VALUE;
		$("#summaryCard .alert").alert('close');

		//workloads type
		if ($("#marginalWorkload").css('display') != 'none'){
			workloadsData.workloadsType = 'marginals';
		} else {
			workloadsData.workloadsType = 'general';
		};

		workloadsData.workloads = workloads;
		function postHDMM1() {
			return $.ajax({
					method: "POST",
					dataType: "json",
					async: true,
					data: JSON.stringify(workloadsData),
					url: "/api/hdmm1",
					success:function(msg){
						$('#hdmmProgress .progress-bar').html(msg.stage);
						hdmm1 = msg;
                        dataFile = hdmm1['data_file_name']
                        codeFile1 = hdmm1['code_file_name']
                        expectedError1 = msg['expected_error_raw']
					},
                    error:function(msg){
                        console.log(msg);
                    }
				});
		};
		function postHDMM2() {
			return $.ajax({
					method: "POST",
					dataType: "json",
					async: true,
					data: JSON.stringify(workloadsData),
					url: "/api/hdmm2",
					success:function(msg){
						$('#hdmmProgress .progress-bar').html(msg.stage);
						hdmm2 = $.extend(true,{},msg);
                        dataFile = hdmm2['data_file']
                        codeFile2 = hdmm2['code']
                        expectedError2 = msg['expected_error']
					},
                    error:function(msg){
                        console.log(msg);
                    }
				});
		};
		function postHDMM3() {
			return $.ajax({
					method: "POST",
					dataType: "json",
					async: true,
					data: JSON.stringify(workloadsData),
					url: "/api/hdmm3",
					success:function(msg){
						$('#hdmmProgress .progress-bar').html(msg.stage);
						hdmm3 = $.extend(true,{},msg);
                        dataFile = hdmm3['data_file'];
                        codeFile3 = hdmm3['code_file'];
                        expectedError3 = msg['Graph']['expected_error'];
					},
                    error:function(msg){
                        console.log(msg);
                    }
				});
		};
		function postIdentity() {
			return $.ajax({
					method: "POST",
					dataType: "json",
					async: true,
					data: JSON.stringify(workloadsData),
					url: "/api/identity",
					success:function(msg){
						$('#hdmmProgress .progress-bar').html(msg.stage);
						identity = $.extend(true,{},msg);
						identity.method = 'Identity'
					},
                    error:function(msg){
                        console.log(msg);
                    }
				});
		};
		if (workloadsData.workloadsType == 'general'){
			if (workloads.length > 1) {
				$.when( postHDMM3(), postIdentity() ).done(function(){
					console.log(hdmm3);
					var hasDensity = false;
					var pdfDeprecated = [];
					for (var index = 0; index < workloads.length; ++index) {
						hdmm3.Table[index].workloadString = workloads[index].workloadString.replace(/\+/g,'⊗');
						hdmm3.Table[index].wid = workloads[index].wid.replace("Row", " ");
						if (hdmm3.Table[index].pdf != false) {
							hasDensity = true;
						} else {
							pdfDeprecated.push(hdmm3.Table[index]);
						};
					};
					MultiWorkloadsSummaryTable(hdmm3.Graph, identity);
					MultiWorkloadsHDMMTable(hdmm3.Table);
					if (hasDensity) {
						WorklodasPDFChart();
						PDFChart();
						$('#wlpdfChart').show();
						$('#pdfChart').show();
						hdmm3.Graph.pdf_x.splice(0, 0, "x");
						hdmm3.Graph.pdf.splice(0, 0, "HDMM");
						identity.pdf_x.splice(0, 0, "x");
						identity.pdf.splice(0, 0, "Identity");
						setTimeout(function(){
							for (var index = 0; index < hdmm3.Table.length; ++index) {
								if (hdmm3.Table[index].pdf){
									var dataName = hdmm3.Table[index].wid;
									hdmm3.Table[index].pdf_x.splice(0, 0, "x");
									hdmm3.Table[index].pdf.splice(0, 0, dataName);
									var colors = {};
									colors[dataName] = colorPattern[index];
									wlpdfChart.load({
										columns: [
											hdmm3.Table[index].pdf_x,
											hdmm3.Table[index].pdf,
										],
										colors: colors
									});
								};
							};
							pdfChart.load({
								columns: [
									hdmm3.Graph.pdf_x,
									hdmm3.Graph.pdf,
								],
								colors: {HDMM:'#1f77b4'}
							});
							pdfChart.load({
								columns: [
									identity.pdf_x,
									identity.pdf,
								],
								colors:{Identity:'#2ca02c'}
							});
						},0); // resize for animation
						if (pdfDeprecated.length != 0) {
							for (var index = 0; index < pdfDeprecated.length; ++index) {
								$("#summaryCard .card-body").prepend('<div class="alert alert-warning" role="alert">' + 'Queries in ' + pdfDeprecated[index].wid + ' have identical error.</div>');
							};
						};
					} else if ("message" in hdmm3.Graph) {
						$("#summaryCard .card-body").prepend('<div class="alert alert-warning" role="alert">' + hdmm3.Graph.message +'</div>');
					}
					else {
						$("#summaryCard .card-body").prepend('<div class="alert alert-warning" role="alert">Queries have identical error, skipping density plots.</div>');
					};
					$('#summaryTable').show();
					$('#hdmmProgress').hide();
					$('#hdmmTable').show();
		            if (dataFile.length > 0)
		            {
		            	$("#exportCard").show();
		            };
		            $("#codeCard").show();
				});
			} else {
				$.when( postHDMM3(), postIdentity() ).done(function(){
					console.log(hdmm3);
					console.log(identity);
					hdmm3.Graph.workloadString = workloads[0].workloadString.replace(/\+/g,'⊗');
					SignleWorkloadSummaryTable(hdmm3.Graph, identity);
					if (hdmm3.Graph.pdf) {
						PDFChart();
						$('#pdfChart').show();
						hdmm3.Graph.pdf_x.splice(0, 0, "x");
						hdmm3.Graph.pdf.splice(0, 0, "HDMM");
						identity.pdf_x.splice(0, 0, "x");
						identity.pdf.splice(0, 0, "Identity");
						console.log(hdmm3.Graph.pdf);
						setTimeout(function(){
							pdfChart.load({
								columns: [
									hdmm3.Graph.pdf_x,
									hdmm3.Graph.pdf,
								],
								colors: {HDMM:'#1f77b4'}
							});
							pdfChart.load({
								columns: [
									identity.pdf_x,
									identity.pdf,
								],
								colors:{Identity:'#2ca02c'}
							});
						},0); // resize for animation
			        } else {
						$("#summaryCard .card-body").prepend('<div class="alert alert-warning" role="alert">Queries have identical error, skipping density plots.</div>');
			        };
			        $('#summaryTable').show();
					$('#hdmmProgress').hide();
		            if (dataFile.length > 0)
		            {
		            	$("#exportCard").show();
		            };
		            $("#codeCard").show();
				});
			}
     	}
	    else {
	    	//marginals results
	    	if (workloads.length > 1) {
				$.when( postHDMM3(), postIdentity() ).done(function(){
					for (var index = 0; index < workloads.length; ++index) {
						hdmm3.Table[index].workloadString = workloads[index].workloadString.replace(/\+/g,'⊗');
						hdmm3.Table[index].wid = workloads[index].wid.replace("Row", " ");
					}
					MultiWorkloadsSummaryTable(hdmm3.Graph, identity);
					MultiWorkloadsHDMMTable(hdmm3.Table);
					$("#summaryCard .card-body").prepend('<div class="alert alert-warning" role="alert">Marginals have idential error, skipping density plots.</div>');
					$('#summaryTable').show();
					$('#hdmmProgress').hide();
					$('#hdmmTable').show();
		            if (dataFile.length > 0)
		            {
		            	$("#exportCard").show();
		            };
		            $("#codeCard").show();
				});
			} else {
				$.when( postHDMM3(), postIdentity() ).done(function(){
					hdmm3.Graph.workloadString = workloads[0].workloadString.replace(/\+/g,'⊗');
					SignleWorkloadSummaryTable(hdmm3.Graph, identity);
					$("#summaryCard .card-body").prepend('<div class="alert alert-warning" role="alert">Marginals have idential error, skipping density plots.</div>');
					$('#summaryTable').show();
					$('#hdmmProgress').hide();
		            if (dataFile.length > 0)
		            {
		            	$("#exportCard").show();
		            };
		            $("#codeCard").show();
				});
			};
	    };

	    $("#workloadBodyWrap").collapse('hide');
		$("#workloadCard #cardStatus").removeClass().addClass('fa fa-check-circle');
		$("#summaryCard").removeClass('card-collapsed').addClass('card-collapse');
		$("#summaryCard").show();
		$('#summaryTable').hide();
		eps = $('#epsilon').val();
	});
	
	$("#summaryCard").on('click', '[id="espRefresh"]', function(){
		eps = $('#epsilon').val();
		$('#hdmmTable').hide();
		$('#summaryTable').hide();
		$('#cdfChart').hide();
		$('#wlpdfChart').hide();
		$("#summaryCard .alert").alert('close');
		var new_hdmm3 = $.extend(true,{}, hdmm3);
		var new_identity = $.extend(true,{}, identity);
		var workloads = workloadsData.workloads;

		if (workloadsData.workloadsType == 'general'){
			new_hdmm3.Graph.expected_error = new_hdmm3.Graph.expected_error/eps;
			new_identity.expected_error = new_identity.expected_error/eps;
			if (workloads.length > 1) {
				var hasDensity = false;
				var pdfDeprecated = [];
				for (var index = 0; index < workloads.length; ++index) {
					if (new_hdmm3.Table[index].pdf != false) {
						hasDensity = true;
						new_hdmm3.Table[index].pdf_x.shift();
						new_hdmm3.Table[index].pdf_x = scalarMultiply(new_hdmm3.Table[index].pdf_x, 1/eps);
					} else {
						pdfDeprecated.push(new_hdmm3.Table[index]);
					};
					new_hdmm3.Table[index].expected_error = new_hdmm3.Table[index].expected_error/eps;
				};
				MultiWorkloadsSummaryTable(new_hdmm3.Graph, new_identity);
				MultiWorkloadsHDMMTable(new_hdmm3.Table);
				if (hasDensity) {
					new_hdmm3.Graph.pdf_x.shift();
					new_hdmm3.Graph.pdf_x = scalarMultiply(new_hdmm3.Graph.pdf_x, 1/eps);
					new_identity.pdf_x.shift();
					new_identity.pdf_x = scalarMultiply(new_identity.pdf_x, 1/eps);
					$('#wlpdfChart').show();
					$('#pdfChart').show();
					new_hdmm3.Graph.pdf_x.splice(0, 0, "x");
					new_identity.pdf_x.splice(0, 0, "x");
					setTimeout(function(){
						wlpdfChart.unload({done: function() {
							for (var index = 0; index < new_hdmm3.Table.length; ++index) {
								if (new_hdmm3.Table[index].pdf){
									var dataName = new_hdmm3.Table[index].wid;
									new_hdmm3.Table[index].pdf_x.splice(0, 0, 'x');
									var colors = {};
									colors[dataName] = colorPattern[index];
									wlpdfChart.load({
										columns: [
											new_hdmm3.Table[index].pdf_x,
											new_hdmm3.Table[index].pdf,
										],
										colors: colors
									});
								};
							};
						}
						});
						pdfChart.unload({done: function() {
							pdfChart.load({
								unload: true,
								columns: [
									new_hdmm3.Graph.pdf_x,
									new_hdmm3.Graph.pdf,
								],
								colors: {HDMM:'#1f77b4'}
							});
							pdfChart.load({
								columns: [
									new_identity.pdf_x,
									new_identity.pdf,
								],
								colors:{Identity:'#2ca02c'}
							});
							}
						});
					},1000); // resize for animation
					if (pdfDeprecated.length != 0) {
						for (var index = 0; index < pdfDeprecated.length; ++index) {
							$("#summaryCard .card-body").prepend('<div class="alert alert-warning" role="alert">' + 'Queries in ' + pdfDeprecated[index].wid + ' have identical error.</div>');
						};
					};
				} else {
					$("#summaryCard .card-body").prepend('<div class="alert alert-warning" role="alert">Queries have identical error, skipping density plots.</div>');
				};
				$('#summaryTable').show();
				$('#hdmmTable').show();
			} else {
				SignleWorkloadSummaryTable(new_hdmm3.Graph, new_identity);
				if (new_hdmm3.Graph.pdf) {
					new_hdmm3.Graph.pdf_x.shift();
					new_hdmm3.Graph.pdf_x = scalarMultiply(new_hdmm3.Graph.pdf_x, 1/eps);
					new_identity.pdf_x.shift();
					new_identity.pdf_x = scalarMultiply(new_identity.pdf_x, 1/eps);
					new_hdmm3.Graph.pdf_x.splice(0, 0, "x");
					new_identity.pdf_x.splice(0, 0, "x");
					setTimeout(function(){
						pdfChart.unload({done: function() {
							pdfChart.load({
								unload: true,
								columns: [
									new_hdmm3.Graph.pdf_x,
									new_hdmm3.Graph.pdf,
								],
								colors: {HDMM:'#1f77b4'}
							});
							pdfChart.load({
								columns: [
									new_identity.pdf_x,
									new_identity.pdf,
								],
								colors:{Identity:'#2ca02c'}
							});
							}
						});
					},1000); // resize for animation
		        } else {
					$("#summaryCard .card-body").prepend('<div class="alert alert-warning" role="alert">Queries have identical error, skipping density plots.</div>');
		        };
		        $('#summaryTable').show();
			};
     	}
	    else {
	    	new_hdmm3.Graph.expected_error = new_hdmm3.Graph.expected_error/eps;
			new_identity.expected_error = new_identity.expected_error/eps;
	    	//marginals results
	    	if (workloads.length > 1) {
				for (var index = 0; index < workloads.length; ++index) {
					new_hdmm3.Table[index].expected_error = new_hdmm3.Table[index].expected_error/eps;
				}
				MultiWorkloadsSummaryTable(new_hdmm3.Graph, new_identity);
				MultiWorkloadsHDMMTable(new_hdmm3.Table);
				$("#summaryCard .card-body").prepend('<div class="alert alert-warning" role="alert">Marginals have idential error, skipping density plots.</div>');
				$('#summaryTable').show();
				$('#hdmmProgress').hide();
				$('#hdmmTable').show();
	            if (dataFile.length > 0)
	            {
	            	$("#exportCard").show();
	            };
	            $("#codeCard").show();
			} else {
				SignleWorkloadSummaryTable(new_hdmm3.Graph, new_identity);
				$("#summaryCard .card-body").prepend('<div class="alert alert-warning" role="alert">Marginals have idential error, skipping density plots.</div>');
				$('#summaryTable').show();
				$('#hdmmProgress').hide();
	            if (dataFile.length > 0)
	            {
	            	$("#exportCard").show();
	            };
	            $("#codeCard").show();
			};
	    };

	});
	
    $("#exportMetaBtn").click(function(){
        var url = Array();
        eps = $('#epsilon').val();

        if ((expectedError1 <= expectedError2) && (expectedError1 <= expectedError3)) {
            url = "/api/export?type=meta&data=" + encodeURIComponent(JSON.stringify({'type': 'hdmm1', 'data_file_name': dataFile, 'code_file_name': codeFile1, 'eps': eps}));
        }
        else if ((expectedError2 <= expectedError1) && (expectedError2 <= expectedError3)) {
            url = "/api/export?type=meta&data=" + encodeURIComponent(JSON.stringify({'type': 'hdmm2', 'data_file_name': dataFile, 'code_file_name': codeFile2, 'eps': eps}));
        }
        else {
            url = "/api/export?type=meta&data=" + encodeURIComponent(JSON.stringify({'type': 'hdmm3', 'data_file_name': dataFile, 'code_file_name': codeFile3, 'eps': eps}));
        }
        window.location=url
    });

    $("#codeBtn").click(function(){
        var url = Array();

        if ((expectedError1 <= expectedError2) && (expectedError1 <= expectedError3)) {
            url = "/api/code?data=" + encodeURIComponent(JSON.stringify({'type': 'hdmm1', 'code_file_name': codeFile1}));
        }
        else if ((expectedError2 <= expectedError1) && (expectedError2 <= expectedError3)) {
            url = "/api/code?data=" + encodeURIComponent(JSON.stringify({'type': 'hdmm2', 'code_file_name': codeFile2}));
        }
        else {
            url = "/api/code?data=" + encodeURIComponent(JSON.stringify({'type': 'hdmm3', 'code_file_name': codeFile3}));
        }

        window.location=url
    });
});


// function for changing table style to DataTable
function domainDataTable(res){
	var table = $('#domainTable').DataTable({
		"info":true,
		"language": {
			"info": "Showing page _PAGE_ of _PAGES_"
		},
		"data":res.data,
		"searching":true,
		"destroy": true,
		"select":true,
		"ordering": false,
		"pageLength": 5,
		"paging": true,
		"lengthChange": false,
		"columns": [{"className":'details-control',
								"orderable":false,
								"data":null,
								"defaultContent": ''},
								{title:"Domain Name",data: "name" },
								{title:"Type",data: "type" },
								{title:"Minimum",data: "min" },
								{title:"Maximum",data: "max" }],
		"order": []
	});

	$('#domainTable tbody').on('click', 'td.details-control', function () {
		var tr = $(this).closest('tr');
		var row = table.row( tr );

		if ( row.child.isShown() ) {
			// This row is already open - close it
			row.child.hide();
			tr.removeClass('shown');
		}
		else {
			// Open this row
			if (row.data().type == "Numerical"){
				row.child(util.domainEditorNumerical
									.replace(/{{form-id}}/g, 'domainEditor-'+row.data().name)
									.replace(/{{lower-bound}}/g, row.data().min)
									.replace(/{{upper-bound}}/g, row.data().max)
									.replace(/{{bucket-size}}/g, Math.ceil((row.data().max - row.data().min)/100))).show();
			}
			else {
				row.child(util.domainEditorCategorical
									.replace(/{{form-id}}/g, 'domainEditor-'+row.data().name)
									.replace(/{{distinct-value}}/g,row.data().value)).show();
			}
			tr.addClass('shown');
			$.fn.selectpicker.Constructor.BootstrapVersion = '4';
			$('.selectpicker').selectpicker({hideDisabled: false});
		}
	});
};

function appendDomainFromTable(domainName, domainType){
	var meta = new Object();
	meta.type = domainType;
	meta.name = domainName;
	meta.buildingBlock = 'total';
	if (domainType == 'Numerical'){
		$("#domainEditor-" + domainName + " input").removeClass('highlight');
		var minimum = $("#domainEditor-" + domainName + " #lowerBound").val();
		var maximum = $("#domainEditor-" + domainName + " #upperBound").val();
		var bucketSize = $("#domainEditor-" + domainName + " #bucketSize").val();
		if (minimum == ''|| minimum >= maximum) {
			$("#domainEditor-" + domainName + " #lowerBound").addClass('highlight');
			return false
		};
		if (maximum == '') {
			$("#domainEditor-" + domainName + " #upperBound").addClass('highlight');
			return false
		};
		if (bucketSize == '' || bucketSize <= 0 || bucketSize > maximum - minimum) {
			$("#domainEditor-" + domainName + " #bucketSize").addClass('highlight');
			return false
		};
		meta.minimum = minimum;
		meta.maximum = maximum;
		meta.bucketSize = bucketSize;

	};
	if (domainType == 'Categorical'){
		$("#domainEditor-" + domainName + " input").removeClass('highlight');
		var distinctValue = $("#domainEditor-" + domainName + " #valueList").val();
		var isOrdered = $("#domainEditor-" + domainName + " #isOrdered").is(':checked');
		if (distinctValue == '') {
			$("#domainEditor-" + domainName + " #valueList").addClass('highlight');
			return false
		};
		meta.isOrdered = isOrdered;
		meta.minimum = 0;
		meta.maximum = distinctValue.split(',').length-1;
		meta.bucketSize = 1;
		meta.values = distinctValue.split(',');
	};
	//insert or update meta data to domainMeta array
	var hasMatch = domainRetrieval(meta.name);

	if (hasMatch){
		domainMeta[hasMatch-1] = meta;
	} else{
		domainMeta.push(meta);
	};
	return true
};

function appendDomainFromModal(domainName, domainType){
	var meta = new Object();
	meta.type = domainType;
	meta.name = domainName;
	meta.buildingBlock = 'total';
	if (domainType == 'Numerical'){
		$("#domainInputNumerical input").removeClass('highlight');
		var minimum = $("#domainInputNumerical #lowerBound").val();
		var maximum = $("#domainInputNumerical #upperBound").val();
		var bucketSize = $("#domainInputNumerical #bucketSize").val();
		if (minimum == '' || minimum >= maximum) {
			$("#domainInputNumerical #lowerBound").addClass('highlight');
			return false
		};
		if (maximum == '') {
			$("#domainInputNumerical #upperBound").addClass('highlight');
			return false
		};
		if (bucketSize == '' || bucketSize <= 0 || bucketSize > maximum - minimum) {
			$("#domainInputNumerical #bucketSize").addClass('highlight');
			return false
		};
		meta.minimum = minimum;
		meta.maximum = maximum;
		meta.bucketSize = bucketSize;
	};
	if (domainType == 'Categorical'){
		$("#domainInputCategorical #valueList").removeClass('highlight')
		var distinctValue = $("#domainInputCategorical #valueList").val();
		var isOrdered = $("#domainInputCategorical #isOrdered").is(':checked');
		if (distinctValue == '') {
			$("#domainInputCategorical #valueList").addClass('highlight');
			return false
		};
		meta.isOrdered = isOrdered;
		meta.minimum = 0;
		meta.maximum = distinctValue.split(',').length - 1;
		meta.bucketSize = 1;
		meta.values = distinctValue.split(',');
	};
	//insert or update meta data to domainMeta array
	var hasMatch = domainRetrieval(meta.name);

	if (hasMatch){
		domainMeta[hasMatch-1] = meta;
	} else{
		domainMeta.push(meta);
	};
	return true
};

function removeDomain(domainName){
	for (var index = 0; index < domainMeta.length; ++index) {
	 var item = domainMeta[index];
	 if(item.name == domainName){
	 	domainMeta.splice(index,1);
	 	break;
	 }
	};
};

function domainRetrieval(domainName){
	for (var index = 0; index < domainMeta.length; ++index) {
		 var item = domainMeta[index];
		 if(item.name == domainName){
		   return index+1
		 }
	};
	return false
};

function domainOptions(selected){
	var options = '';
	for (var index = 0; index < domainMeta.length; ++index) {
		var item = domainMeta[index];
		options += util.domainPicker.replace(/{{domain-name}}/g,item.name) + '\n';
	};
	return options
};

function domainCheckbox(selected){
	var options = '';
	var perLine = 5;
	var optionsLine = '';
	console.log(domainMeta.length);
	for (var index = 0; index < domainMeta.length; ++index) {
		var item = domainMeta[index];
		console.log(item.name)
		optionsLine += util.domainCheckbox.replace(/{{domain-name}}/g,item.name) + '\n';
		if ((index+1) % perLine == 0){
			options += '<div class="form-row">' + optionsLine + '</div>'
			optionsLine = '';
		}
	};
	if (optionsLine != ''){
		options += '<div class="form-row">' + optionsLine + '</div>'
	};
	console.log(options);
	return options
};

function blockString (type, domain, name="") {
	if (type == 'identity'){
		return 'I(' + domain + ')'
	};
	if (type == 'prefix'){
		return 'P(' + domain + ')'
	};
	if (type == 'allrange'){
		return 'R(' + domain + ')'
	};
	if (type == 'total'){
		return 'T(' + domain + ')'
	};
	if (type == 'customized'){
		return 'C(' + domain + ')' + '[' + name + ']'
	};
};

function reverseBlockString(string) {
	if (string[0] == 'I'){
		return 'identity'
	};
	if (string[0] == 'P'){
		return 'prefix'
	};
	if (string[0] == 'R'){
		return 'allrange'
	};
	if (string[0] == 'T'){
		return 'total'
	};
};

function distinctDomain ()	{
	var selected = selectedDomain();
	$("#workloadCard form[id^=buildingBlock]").each(function () {
		var bid = $(this).closest('form').attr('id');
		var domain = $('#' + bid + ' #blockDomain option:selected').val();
		$('#' + bid + ' #blockDomain option').each(function()
		{
		    $(this).prop('disabled', false);
		});
		for (var index = 0; index < selected.length; ++index) {
			if (selected[index] != domain){
				$('#' + bid + ' #blockDomain' + ' option[value=' + selected[index] + ']').prop('disabled', 'disabled');
			}
		}
	});
	$('.selectpicker').selectpicker({hideDisabled: false});
};

function selectedDomain ()	{
	var selected = new Array();
	$("#workloadCard form[id^=buildingBlock]").each(function () {
		var bid = $(this).closest('form').attr('id');
		var domain = $('#' + bid + ' #blockDomain option:selected').val();
		if (domain != "") {
			selected.push(domain);
		};
	});
	return selected;
};

function romanize (num) {
	if (!+num)
		return false;
	var	digits = String(+num).split(""),
		key = ["","C","CC","CCC","CD","D","DC","DCC","DCCC","CM",
		       "","X","XX","XXX","XL","L","LX","LXX","LXXX","XC",
		       "","I","II","III","IV","V","VI","VII","VIII","IX"],
		roman = "",
		i = 3;
	while (i--)
		roman = (key[+digits.pop() + (i * 10)] || "") + roman;
	return Array(+digits.join("") + 1).join("M") + roman;
};

function optChart(data) {
	lossChart = c3.generate({
		bindto:'#lossChart',
	    data: {
	    	x: 'x',
	        columns: [],
        },
    	axis: {
	        x: {
	        	label: 'Training Epoch'
	        },
	        y: {
            	label: 'Log Loss',
            	padding: {top: 0, bottom: 0}
        	}
    	},
    	title: {
          text: 'Optimization Procedure',
        },
        legend: {
          show: true,
        },
        tooltip: {
          format: {
            title: function (index) {return 'Epoch = ' + index},
          }
        },
        zoom: {
			enable: true,
			rescale: true,
			type: 'drag'
		}
	});
};

function CDFChart(data) {
	cdfChart = c3.generate({
		bindto:'#cdfChart',
	    data: {
	    	type: 'area-spline',
	    	x: 'x',
	        columns: []
        },
    	axis: {
	        x: {
	        	label: 'Per Query Root Error',
	        	tick: {
	        		format: d3.format('.3f'),
	        		count: 5
	        	}
	        },
	        y: {
            	label: 'Cumulative Distribution',
            	padding: {top: 10, bottom: 0}
        	}
    	},
    	title: {
          text: 'Query error distribution',
        },
        legend: {
          show: true,
        },
        point: {
        	show: false
    	},
        tooltip: {
          format: {
            title: function (index) {return 'Error = ' + Math.round(index * 100) / 100},
            value: function (value, ratio, id, index) {
                return "CDF = " + Math.round(value * 100) / 100;
            }
          }
        },
		zoom: {
			enable: true,
			rescale: true,
			type: 'drag'
		}
	});
};

function WorklodasPDFChart(data) {
	wlpdfChart = c3.generate({
		bindto:'#wlpdfChart',
	    data: {
	    	type: 'area-spline',
	    	x: 'x',
	        columns: []
        },
    	axis: {
	        x: {
	        	label: 'Per Query Root Error',
	        	tick: {
	        		format: d3.format('.3f'),
	        		count: 10
	        	}
	        },
	        y: {
            	label: 'Probability Density',
            	padding: {top: 10, bottom: 10}
        	}
    	},
    	title: {
          text: 'Query error density for each workload',
        },
        legend: {
          show: true,
        },
        point: {
        	show: false
    	},
        tooltip: {
          format: {
            title: function (index) {return 'Error = ' + Math.round(index * 100) / 100},
            value: function (value, ratio, id, index) {
                return "PDF = " + Math.round(value * 100) / 100;
            }
          }
        },
		zoom: {
			enable: true,
			rescale: true,
			type: 'drag'
		}
	});
};

function PDFChart(data) {
	pdfChart = c3.generate({
		bindto:'#pdfChart',
	    data: {
	    	type: 'area-spline',
	    	x: 'x',
	        columns: [],
        },
    	axis: {
	        x: {
	        	label: 'Per Query Root Error',
	        	tick: {
	        		format: d3.format('.3f'),
	        		count: 10
	        	}
	        },
	        y: {
            	label: 'Probability Density Function',
            	padding: {top: 10, bottom: 10}
        	}
    	},
    	title: {
          text: 'Query error density',
        },
        legend: {
          show: true,
        },
        point: {
        	show: false
    	},
        tooltip: {
          format: {
            title: function (index) {return 'Error = ' + Math.round(index * 100) / 100},
            value: function (value, ratio, id, index) {
                return "PDF = " + Math.round(value * 100) / 100;
            }
          }
        },
		zoom: {
			enable: true,
			rescale: true,
			type: 'drag'
		}
	});
};

function BarChart(data) {
	barChart = c3.generate({
		bindto:'#cdfChart',
	    data: {
	    	type: 'bar',
	    	x: 'x',
	        columns: []
        },
        bar: {
        	width: 40
        },
    	axis: {
	        x: {
	        	label: 'Workloads',
	        	type: 'category',
	        },
	        y: {
            	label: 'Expected Root MSE',
            	padding: {top: 10, bottom: 0}
        	}
    	},
    	title: {
          text: 'Marginals root MSE',
        },
        legend: {
          show: true,
        },
        tooltip: {
          format: {
            value: function (value, ratio, id, index) {
                return Math.round(value * 100) / 100;
            }
          }
        }
	});
};

function MultiWorkloadsHDMMTable(data){
	var summaryRows = '';
	for (var index = 0; index < data.length; ++index) {
		summaryRows += '<tr><td style="vertical-align : middle;text-align:center;">'+ data[index].wid.replace("Row", " ")
						 + '</td><td>' + data[index].workloadString.replace(/\+/g," ⊗ ")
						 + '</td><td style="vertical-align : middle;text-align:center;">' + data[index].num_query
						 + '</td><td style="vertical-align : middle;text-align:center;">' + data[index].expected_error.toFixed(4)
						 + '</td></tr>';
	};
	$('#hdmmTable').show();
	$('#hdmmTable').empty();
	$('#hdmmTable').append(util.multiWorkloadsHDMMTable.replace(/{{summaryRows}}/g, summaryRows));
};

function MultiWorkloadsSummaryTable(hdmm,ide){
	$('#summaryTable').show();
	$('#summaryTable').empty();
	$('#summaryTable').append(util.multiWorkloadsSummaryTable.replace(/{{hdmmError}}/g, hdmm.expected_error.toFixed(4))
												.replace(/{{ideError}}/g, ide.expected_error.toFixed(4)));
};

function SignleWorkloadSummaryTable(hdmm,ide){
	$('#summaryTable').show();
	$('#summaryTable').empty();
	$('#summaryTable').append(util.singleWorkloadSummaryTable.replace(/{{workloadString}}/g, hdmm.workloadString)
												.replace(/{{queryNumber}}/g, hdmm.num_query)
												.replace(/{{hdmmError}}/g, hdmm.expected_error.toFixed(4))
												.replace(/{{ideError}}/g, ide.expected_error.toFixed(4)));
};

function range(start, end, step){
    var array = new Array();
    for(var i = start; i <= end; i+=step)
    {
        array.push(i);
    };
    if(array[array.length-1] != end){
    	array.push(end);
    };
    return array;
};

function indexFinder(arr, key, value){
	var index = arr.findIndex(function(item, i){
			return item[key] == value
	});
	return index;
};

function customizedParser(input, values, domain){
	var index = domainMeta.findIndex(function(item, i){
		return item.name == domain
	}),
	domainType = domainMeta[index].type,
	parser = new Array();
	for (var idx = 0; idx < input.length; ++idx) {
		if(domainType == "Categorical"){
			if (values.indexOf(input[idx]) != -1){
				parser.push(input[idx]);
			} else {
				return false;
			}
		} else {
			if (input[idx].indexOf("-") != -1){
				var subrange = range(parseInt(input[idx].split("-")[0]),
									 parseInt(input[idx].split("-")[1]),
								     parseInt(domainMeta[index].bucketSize));
				for (var i = 0; i < subrange.length; ++i) {
					if (values.indexOf(subrange[i].toString()) != -1){
						parser.push(subrange[i]);
					} else {
						return false;
					};
				};
			} else {
				if (values.indexOf(input[idx]) != -1){
					parser.push(parseInt(input[idx]));
				} else {
					return false;
				}
			}
		}
	};
	return parser;
};

function sizeCalculator(){
	
	for (var idx = 0; idx < workloadsObj.length; ++idx) {
		var tol = 1
		for (var jdx = 0; jdx < workloadsObj[idx].length; ++jdx){
			var buildingBlock = workloadsObj[idx][jdx].buildingBlock;
			var basic = Math.ceil((workloadsObj[idx][jdx].maximum - workloadsObj[idx][jdx].minimum)/workloadsObj[idx][jdx].bucketSize) + 1;
			// if (buildingBlock == "allrange"){
			// 	tol *= Math.ceil(((1+basic)*basic)/2)
			// } else 
			if (buildingBlock == "total"){
				tol *= 1
			} else {
				tol *= basic
			};
		};
		if (tol > 5e5){
			return true
		};
	};
	return false
};

function scalarMultiply(arr, multiplier) {
   for (var i = 0; i < arr.length; i++)
   {
      arr[i] *= multiplier;
   }
   return arr;
}




